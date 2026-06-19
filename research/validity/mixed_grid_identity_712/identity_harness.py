#!/usr/bin/env python
"""PR #712 - mixed-grid Route-B greedy-identity differential (in-memory fake-quant, NO disk writes).

QUESTION (launch contract leg): does the realizable Route-B recovery config
  - 40 per_layer_input_gate (PLIG) promoted to group_size 32, AND
  - the whole fused qkv block promoted to g32 for every layer holding one of ubel #700's
    targeted q/k/v  (servable set: L0,L1,L18 -> {q,k,v}; L40,L41 -> {q}  => 11 attn modules)
preserve strict-#319 byte-exact greedy-token identity vs the locked g128 anchor
(submission int4_g128_lmhead)?  If a *realizable* mixed-grid recovery flips any greedy
token, the int4-body recovery arm is dead on the launch contract regardless of AIME quality.

METHOD (disk-safe, HBM only):
  anchor  = dequant_g128 dense bf16 weights  (run_compressed=False load == served grid).
  RouteB  = anchor with target modules re-gridded to g32.  We do NOT have qat_unq on disk
            (only its g128 projection), so the *faithful* recovery delta is modeled exactly:
            quant_g32(qat_unq) ~= qat_unq  and  anchor = qat_unq - r, where r is the g128
            quantization residual r ~ Uniform[-s/2, +s/2] (s = on-disk g128 group scale).
            => Route-B - anchor ~= r.  We inject  d = alpha * U(-s/2, s/2)  per element
            (alpha=1 == faithful magnitude+distribution), swept over seeds, AND cross-check
            with the PR-literal structured requant_g32(dequant_g128) (note: requant carries a
            ~6.7% scale-re-derivation inflation -> it is a CONSERVATIVE/pessimistic proxy).

GREEDY DIFFERENTIAL: argmax(RouteB logits) vs argmax(anchor logits) per position over the
gate's 128 sharegpt prompts (seed=1) + an anchor greedy continuation.  The HF-vs-served-Marlin
kernel difference CANCELS (anchor and RouteB share this one forward); a clean differential =>
RouteB served greedy tokens == g128 served greedy tokens => RouteB passes the 128/128 gate by
transitivity (g128 passes today). Marlin int4 GEMM is deterministic across the g32/g128 group
sizes (my PR #680: maxdiff=0), so a clean weight differential is a strong PRESERVED verdict.

NO checkpoint build, NO disk writes to any served path, NO HF job, NO submission.
"""
from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # inherited =7 is stale; force assigned A10G
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import argparse, json, random, sys, time
from pathlib import Path
import torch

CKPT = "/workspace/gemma_build/int4_g128_lmhead"
DATASET = ("/workspace/senpai/target/official/main_bucket/shared_resources/"
           "speed_benchmark/data/eval_prompts_sharegpt.json")
HERE = Path("/workspace/senpai/target/research/validity/mixed_grid_identity_712")

# ---- Route-B realizable target sets (see ubel700_subset48.json + #708 servability) ----
PLIG_LAYERS = [23,0,1,22,29,8,21,14,17,20,15,13,28,11,16,18,12,24,6,19,27,5,41,9,4,
               10,26,33,35,34,36,32,30,37,7,31,25,38,39,3]            # 40 PLIG (ubel pick)
QKV_FULL_LAYERS = [0, 1, 18]      # own k/v -> whole fused block {q,k,v} promoted to g32
QKV_QONLY_LAYERS = [40, 41]       # KV-shared -> only q_proj exists (standalone g32)

def plig_targets():
    return [f"model.language_model.layers.{L}.per_layer_input_gate" for L in PLIG_LAYERS]

def qkv_targets():
    t = []
    for L in QKV_FULL_LAYERS:
        for p in ("q_proj", "k_proj", "v_proj"):
            t.append(f"model.language_model.layers.{L}.self_attn.{p}")
    for L in QKV_QONLY_LAYERS:
        t.append(f"model.language_model.layers.{L}.self_attn.q_proj")
    return t                                                          # 11 attn modules

def config_targets(name):
    if name == "plig40":  return plig_targets()
    if name == "qkv11":   return qkv_targets()
    if name == "both51":  return plig_targets() + qkv_targets()
    raise ValueError(name)


def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.utils.quantization_config import CompressedTensorsConfig
    tok = AutoTokenizer.from_pretrained(CKPT)
    model = AutoModelForCausalLM.from_pretrained(
        CKPT, dtype=torch.bfloat16,
        quantization_config=CompressedTensorsConfig(run_compressed=False))
    model.eval().to("cuda:0")
    return model, tok


def snapshot_anchor(model, names):
    """For each target module: anchor weight (CPU bf16) + per-element g128 scale (CPU f32)."""
    snap = {}
    for n in names:
        mod = model.get_submodule(n)
        W = mod.weight.detach()
        sc = mod.weight_scale.detach().float()              # (out, ng)
        out_dim, in_dim = W.shape
        ng = sc.shape[1]
        gs = in_dim // ng
        assert in_dim == ng * gs, f"{n}: in_dim {in_dim} not divisible by ng {ng}"
        scale_full = sc.repeat_interleave(gs, dim=1)        # (out, in)
        snap[n] = {"W": W.clone().cpu(), "scale_full": scale_full.cpu(),
                   "gs128": gs, "in_dim": in_dim, "out_dim": out_dim}
    return snap


def requant_g32(W, gs=32):
    """Structured PR-literal re-grid: dequant_g32(quant_g32(W)) on a dense weight (f32)."""
    from compressed_tensors.quantization import QuantizationArgs
    from compressed_tensors.quantization.lifecycle.forward import quantize, dequantize
    from compressed_tensors.quantization.utils.helpers import calculate_qparams
    qa = QuantizationArgs(num_bits=4, type="int", strategy="group",
                          group_size=gs, symmetric=True, observer="minmax")
    W = W.float()
    out_dim, in_dim = W.shape
    ng = in_dim // gs
    wg = W.reshape(out_dim, ng, gs)
    scale, zp = calculate_qparams(wg.amin(-1), wg.amax(-1), qa)
    q = quantize(W, scale, zp, qa)
    return dequantize(q, scale, zp, qa).float()


@torch.no_grad()
def set_config(model, snap, names, mode, alpha=1.0, seed=0):
    """Overwrite each target module .weight in place per the chosen Route-B model.
    mode: 'anchor' (unperturbed), 'uniform' (alpha*U(-s/2,s/2)), 'requant_g32'."""
    g = torch.Generator(device="cuda:0"); g.manual_seed(seed)
    stats = {"n_mod": 0, "rel_frob_mean": 0.0}
    rels = []
    for n in names:
        s = snap[n]
        W = s["W"].to("cuda:0").float()
        if mode == "anchor":
            newW = W
        elif mode == "uniform":
            sf = s["scale_full"].to("cuda:0")
            u = (torch.rand(W.shape, generator=g, device="cuda:0") - 0.5)  # U(-0.5,0.5)
            newW = W + alpha * u * sf                                       # U(-s/2, s/2)*alpha
        elif mode == "requant_g32":
            if s["in_dim"] % 32 != 0:
                newW = W  # cannot g32-regrid (in_dim not /32); leave anchor (rare; report)
            else:
                newW = requant_g32(W, 32).to("cuda:0")
        else:
            raise ValueError(mode)
        if mode != "anchor":
            rels.append(float((newW - W).norm() / W.norm().clamp_min(1e-12)))
        model.get_submodule(n).weight.data.copy_(newW.to(torch.bfloat16))
        stats["n_mod"] += 1
    if rels:
        stats["rel_frob_mean"] = sum(rels) / len(rels)
        stats["rel_frob_max"] = max(rels)
    return stats


def load_prompts(n, seed=1):
    data = json.loads(Path(DATASET).read_text())
    recs = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        conv = item.get("conversations")
        if not isinstance(conv, list) or len(conv) < 2 or not isinstance(conv[0], dict):
            continue
        p = conv[0].get("value")
        if isinstance(p, str) and p:
            recs.append({"id": str(item.get("id", index)), "prompt_text": p})
    random.Random(seed).shuffle(recs)
    return recs[:n]


@torch.no_grad()
def tok_prompt(tok, text):
    enc = tok.apply_chat_template([{"role": "user", "content": text}],
                                  add_generation_prompt=True, tokenize=True,
                                  return_tensors="pt", return_dict=True)
    return enc["input_ids"][0].to("cuda:0")        # (P,) text-only


@torch.no_grad()
def greedy_continuation(model, prompt_ids, C):
    """B=1 greedy decode C tokens; returns list[int] continuation ids."""
    ids = prompt_ids.unsqueeze(0)
    past, cont = None, []
    cur = ids
    for _ in range(C):
        out = model(input_ids=cur, past_key_values=past, use_cache=True)
        past = out.past_key_values
        nxt = int(out.logits[0, -1].argmax())
        cont.append(nxt)
        cur = torch.tensor([[nxt]], device="cuda:0")
        del out
    return cont


@torch.no_grad()
def argmax_over(model, full_ids):
    """argmax per position for sequence full_ids (1D). returns (T,) long on cuda."""
    out = model(input_ids=full_ids.unsqueeze(0), use_cache=False)
    am = out.logits[0].argmax(-1)
    del out
    return am


@torch.no_grad()
def anchor_margin(model, full_ids, plen):
    """Top1-top2 logit gap at each continuation predicting-position (plen-1 .. T-2)."""
    out = model(input_ids=full_ids.unsqueeze(0), use_cache=False)
    lg = out.logits[0].float()
    top2 = lg.topk(2, dim=-1).values
    margin = (top2[:, 0] - top2[:, 1])
    del out
    return margin


def run(args):
    t0 = time.time()
    model, tok = load_model()
    print(f"[load] {time.time()-t0:.1f}s", flush=True)
    prompts = load_prompts(args.num_prompts, seed=1)
    print(f"[prompts] n={len(prompts)} C={args.cont}", flush=True)

    # tokenize + anchor greedy continuations (the faithful greedy contexts)
    all_names = sorted(set(plig_targets() + qkv_targets()))
    snap = snapshot_anchor(model, all_names)
    set_config(model, snap, all_names, "anchor")           # ensure clean anchor weights
    seqs = []          # per prompt: dict(plen, full_ids, ref_cont, margin)
    tgen = time.time()
    for i, rec in enumerate(prompts):
        pid = tok_prompt(tok, rec["prompt_text"])
        if pid.numel() > args.max_plen:
            pid = pid[-args.max_plen:]                       # cap very long prompts
        cont = greedy_continuation(model, pid, args.cont)
        full = torch.cat([pid, torch.tensor(cont, device="cuda:0")])
        seqs.append({"plen": int(pid.numel()), "full": full,
                     "ref_cont": cont, "id": rec["id"]})
        if (i + 1) % 16 == 0:
            print(f"  [anchor-gen] {i+1}/{len(prompts)} ({time.time()-tgen:.0f}s)", flush=True)
    # anchor continuation margins (tie analysis) on the continuation predicting-positions
    for s in seqs:
        m = anchor_margin(model, s["full"], s["plen"])
        # predicting-position p predicts token p+1; continuation tokens are at [plen .. T-1]
        s["cont_margin"] = m[s["plen"]-1: s["full"].numel()-1].detach().cpu()
    print(f"[anchor-gen+margin] {time.time()-tgen:.0f}s", flush=True)

    def eval_config(cfg, mode, alpha, seed):
        names = config_targets(cfg)
        st = set_config(model, snap, names, mode, alpha=alpha, seed=seed)
        n_cont = 0; n_flip = 0; first_flips = []; flip_margins = []
        prompts_diverged = 0
        for s in seqs:
            am = argmax_over(model, s["full"])             # (T,)
            plen, T = s["plen"], s["full"].numel()
            # predicting-position p (plen-1 .. T-2) predicts token at p+1
            pred = am[plen-1: T-1]                          # (C,)
            ref = torch.tensor(s["ref_cont"], device="cuda:0")
            mism = (pred != ref)
            n_cont += int(ref.numel())
            nf = int(mism.sum())
            n_flip += nf
            if nf:
                prompts_diverged += 1
                fp = int(mism.float().argmax())            # first flip index in continuation
                first_flips.append(fp)
                flip_margins.extend(s["cont_margin"][mism.cpu()].tolist())
            del am
        set_config(model, snap, names, "anchor")           # restore
        return {
            "cfg": cfg, "mode": mode, "alpha": alpha, "seed": seed,
            "n_cont_positions": n_cont, "n_flips": n_flip,
            "flip_rate": n_flip / max(1, n_cont),
            "prompts_diverged": prompts_diverged, "prompts_total": len(seqs),
            "first_flip_min": (min(first_flips) if first_flips else None),
            "first_flip_median": (sorted(first_flips)[len(first_flips)//2] if first_flips else None),
            "flip_margin_p50": (float(torch.tensor(flip_margins).median()) if flip_margins else None),
            "flip_margin_max": (float(max(flip_margins)) if flip_margins else None),
            "rel_frob_mean": st.get("rel_frob_mean"), "rel_frob_max": st.get("rel_frob_max"),
        }

    results = []
    def add(r):
        results.append(r)
        print(f"  [{r['cfg']:7s} {r['mode']:11s} a={r['alpha']:.2f} s={r['seed']}] "
              f"flip_rate={r['flip_rate']:.3e} flips={r['n_flips']}/{r['n_cont_positions']} "
              f"prompts_div={r['prompts_diverged']}/{r['prompts_total']} "
              f"relF={r['rel_frob_mean']}", flush=True)

    # sanity: anchor-vs-anchor differential must be exactly 0
    add({**eval_config("both51", "anchor", 0.0, 0)})

    if args.debug:
        add(eval_config("both51", "uniform", 1.0, 0))
        json.dump(results, open(HERE / "identity_debug.json", "w"), indent=2)
        print("DEBUG_OK", flush=True); return

    # HEADLINE faithful alpha=1 (multi-seed) for the three ablations
    for cfg in ("both51", "plig40", "qkv11"):
        for seed in range(args.seeds):
            add(eval_config(cfg, "uniform", 1.0, seed))
    # structured PR-literal cross-check (conservative)
    for cfg in ("both51", "plig40", "qkv11"):
        add(eval_config(cfg, "requant_g32", 1.0, 0))
    # magnitude sensitivity curve on the realizable both51 (locate where flips turn on)
    for alpha in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0):
        for seed in range(max(3, args.seeds)):
            add(eval_config("both51", "uniform", alpha, seed))

    out = {"results": results, "meta": {
        "num_prompts": len(seqs), "cont": args.cont, "seeds": args.seeds,
        "plig_modules": len(plig_targets()), "qkv_modules": len(qkv_targets()),
        "both_modules": len(config_targets("both51")),
    }}
    json.dump(out, open(HERE / "identity_results.json", "w"), indent=2)
    print(f"[done] {time.time()-t0:.0f}s -> identity_results.json", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-prompts", type=int, default=128)
    ap.add_argument("--cont", type=int, default=128)
    ap.add_argument("--max-plen", type=int, default=1024)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--debug", action="store_true")
    run(ap.parse_args())
