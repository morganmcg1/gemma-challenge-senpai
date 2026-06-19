#!/usr/bin/env python
"""PR #712 v2 - mixed-grid Route-B greedy-identity, CORRECTED reference (in-memory fake-quant).

v1 BUG (research/validity/mixed_grid_identity_712/identity_results.json): it compared each
config's TEACHER-FORCED prefill argmax (M=T) against the anchor's AUTOREGRESSIVE decode tokens
(M=1, ref_cont). Those two kernel paths differ by the prefill-vs-decode flash split-KV attention
reduction even at ZERO weight change, so anchor-vs-anchor showed a 77-flip near-tie jitter floor
(margin <= 0.375) instead of 0. That floor contaminated every config's flip count.

v2 FIX (= the PR's literal method (a)): compare argmax(routeB) vs argmax(anchor) with BOTH sides
teacher-forced on the same context path. anchor-vs-anchor is then EXACTLY 0 (identical weights,
deterministic prefill), so every residual flip is PURELY the Route-B weight change.
Plus method (b): generate routeB's own greedy decode and compare to the anchor's served decode
stream (ref_cont) token-by-token with early-stop at first divergence -- the faithful served-gate test.

anchor  = dequant_g128 dense bf16 (run_compressed=False == served grid). Underlying weights are
qat_unq (build_quant.py --src /workspace/gemma_build/qat_unq, NOT on disk). The realizable Route-B
recovery is quant_g32(qat_unq); we proxy its delta two ways:
  - requant_g32 : structured quant_g32(dequant_g128) -- a CONCRETE lower-bracket (delta ~ s32/2).
  - uniform a=1 : inject a*U(-s128/2, s128/2) -- the FAITHFUL magnitude of the true delta r128.
The HF-vs-served-Marlin kernel CANCELS in the differential (both arms one shared forward); int4
g128/g32 Marlin GEMM is byte-identical across width (PR #680), so a clean weight differential
transfers to the served gate. NO disk writes to any served path, NO HF job, NO submission.
"""
from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import argparse, json, random, time
from pathlib import Path
import torch

CKPT = "/workspace/gemma_build/int4_g128_lmhead"
DATASET = ("/workspace/senpai/target/official/main_bucket/shared_resources/"
           "speed_benchmark/data/eval_prompts_sharegpt.json")
HERE = Path("/workspace/senpai/target/research/validity/mixed_grid_identity_712")

PLIG_LAYERS = [23,0,1,22,29,8,21,14,17,20,15,13,28,11,16,18,12,24,6,19,27,5,41,9,4,
               10,26,33,35,34,36,32,30,37,7,31,25,38,39,3]            # 40 PLIG (ubel #700)
QKV_FULL_LAYERS = [0, 1, 18]      # own k/v -> whole fused {q,k,v} promoted to g32
QKV_QONLY_LAYERS = [40, 41]       # KV-shared -> only q_proj exists


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
    snap = {}
    for n in names:
        mod = model.get_submodule(n)
        W = mod.weight.detach()
        sc = mod.weight_scale.detach().float()
        out_dim, in_dim = W.shape
        ng = sc.shape[1]
        gs = in_dim // ng
        assert in_dim == ng * gs, f"{n}: in_dim {in_dim} not divisible by ng {ng}"
        scale_full = sc.repeat_interleave(gs, dim=1)
        snap[n] = {"W": W.clone().cpu(), "scale_full": scale_full.cpu(),
                   "gs128": gs, "in_dim": in_dim, "out_dim": out_dim}
    return snap


def requant_g32(W, gs=32):
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
    g = torch.Generator(device="cuda:0"); g.manual_seed(seed)
    rels = []
    for n in names:
        s = snap[n]
        W = s["W"].to("cuda:0").float()
        if mode == "anchor":
            newW = W
        elif mode == "uniform":
            sf = s["scale_full"].to("cuda:0")
            u = (torch.rand(W.shape, generator=g, device="cuda:0") - 0.5)
            newW = W + alpha * u * sf
        elif mode == "requant_g32":
            newW = W if (s["in_dim"] % 32 != 0) else requant_g32(W, 32).to("cuda:0")
        else:
            raise ValueError(mode)
        if mode != "anchor":
            rels.append(float((newW - W).norm() / W.norm().clamp_min(1e-12)))
        model.get_submodule(n).weight.data.copy_(newW.to(torch.bfloat16))
    return (sum(rels) / len(rels)) if rels else 0.0


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
    return enc["input_ids"][0].to("cuda:0")


@torch.no_grad()
def greedy_decode(model, prompt_ids, C, ref=None):
    """B=1 greedy decode up to C tokens. If ref is given, EARLY-STOP at first divergence.
    Returns (tokens, first_div) where first_div is the index of the first token != ref (or -1)."""
    past, cont = None, []
    cur = prompt_ids.unsqueeze(0)
    first_div = -1
    for i in range(C):
        out = model(input_ids=cur, past_key_values=past, use_cache=True)
        past = out.past_key_values
        nxt = int(out.logits[0, -1].argmax())
        cont.append(nxt)
        del out
        if ref is not None and i < len(ref) and nxt != ref[i]:
            first_div = i
            break
        cur = torch.tensor([[nxt]], device="cuda:0")
    return cont, first_div


@torch.no_grad()
def tf_argmax_and_margin(model, full_ids, plen):
    """Teacher-forced argmax + top1-top2 logit margin at continuation predicting-positions.
    predicting-position p (plen-1 .. T-2) predicts token p+1 (the continuation tokens)."""
    out = model(input_ids=full_ids.unsqueeze(0), use_cache=False)
    lg = out.logits[0]                                   # (T, V)
    am = lg.argmax(-1)                                   # (T,)
    T = full_ids.numel()
    am_cont = am[plen-1: T-1].clone()                    # (C,) argmax at cont predicting-pos
    top2 = lg[plen-1: T-1].float().topk(2, dim=-1).values
    margin = (top2[:, 0] - top2[:, 1]).clone()           # (C,)
    del out, lg, am
    return am_cont, margin


def run(args):
    t0 = time.time()
    model, tok = load_model()
    print(f"[load] {time.time()-t0:.1f}s", flush=True)
    prompts = load_prompts(args.num_prompts, seed=1)
    print(f"[prompts] n={len(prompts)} C={args.cont}", flush=True)

    all_names = sorted(set(plig_targets() + qkv_targets()))
    snap = snapshot_anchor(model, all_names)
    set_config(model, snap, all_names, "anchor")

    # ---- Phase 1: anchor served decode (ref_cont) + anchor teacher-forced argmax/margin ----
    seqs = []
    tgen = time.time()
    for i, rec in enumerate(prompts):
        pid = tok_prompt(tok, rec["prompt_text"])
        if pid.numel() > args.max_plen:
            pid = pid[-args.max_plen:]
        ref_cont, _ = greedy_decode(model, pid, args.cont)            # anchor served stream
        full = torch.cat([pid, torch.tensor(ref_cont, device="cuda:0")])
        anchor_tf, margin = tf_argmax_and_margin(model, full, int(pid.numel()))
        seqs.append({"plen": int(pid.numel()), "full": full, "ref_cont": ref_cont,
                     "anchor_tf": anchor_tf.cpu(), "margin": margin.cpu(), "id": rec["id"]})
        if (i + 1) % 16 == 0:
            print(f"  [anchor-gen] {i+1}/{len(prompts)} ({time.time()-tgen:.0f}s)", flush=True)
    print(f"[phase1] {time.time()-tgen:.0f}s", flush=True)

    MARGIN_NEARTIE = 0.5    # logit gap; flips with anchor margin > this are CONFIDENT (not near-tie)

    # ---- Phase 2: method (a) corrected -- routeB TF argmax vs anchor TF argmax ----
    @torch.no_grad()
    def eval_a(cfg, mode, alpha, seed):
        names = config_targets(cfg)
        relf = set_config(model, snap, names, mode, alpha=alpha, seed=seed)
        n_flip = 0; first_flips = []; flip_margins = []; prompts_div = 0
        confident_flips = 0
        for s in seqs:
            am, _m = tf_argmax_and_margin(model, s["full"], s["plen"])
            am = am.cpu()
            ref = s["anchor_tf"]                                       # ANCHOR TF argmax (not decode)
            mism = (am != ref)
            nf = int(mism.sum())
            n_flip += nf
            if nf:
                prompts_div += 1
                first_flips.append(int(mism.float().argmax()))
                fm = s["margin"][mism]
                flip_margins.extend(fm.tolist())
                confident_flips += int((fm > MARGIN_NEARTIE).sum())
        set_config(model, snap, names, "anchor")
        C = seqs[0]["anchor_tf"].numel()
        npos = len(seqs) * C
        return {
            "phase": "a_tf_vs_tf", "cfg": cfg, "mode": mode, "alpha": alpha, "seed": seed,
            "n_cont_positions": npos, "n_flips": n_flip, "flip_rate": n_flip / npos,
            "confident_flips": confident_flips, "confident_rate": confident_flips / npos,
            "prompts_diverged": prompts_div, "prompts_total": len(seqs),
            "first_flip_min": (min(first_flips) if first_flips else None),
            "first_flip_median": (sorted(first_flips)[len(first_flips)//2] if first_flips else None),
            "flip_margin_p50": (float(torch.tensor(flip_margins).median()) if flip_margins else None),
            "flip_margin_max": (float(max(flip_margins)) if flip_margins else None),
            "rel_frob_mean": relf,
        }

    results = []
    def addA(cfg, mode, alpha, seeds):
        for sd in seeds:
            r = eval_a(cfg, mode, alpha, sd)
            results.append(r)
            print(f"  A[{cfg:7s} {mode:11s} a={alpha:.2f} s={sd}] flips={r['n_flips']}/{r['n_cont_positions']} "
                  f"({r['flip_rate']:.3e}) confident={r['confident_flips']} "
                  f"prompts_div={r['prompts_diverged']}/{r['prompts_total']} "
                  f"mmax={r['flip_margin_max']} relF={r['rel_frob_mean']:.4f}", flush=True)

    addA("both51", "anchor", 0.0, [0])                 # MUST be exactly 0
    addA("both51", "requant_g32", 1.0, [0])            # concrete realizable
    addA("plig40", "requant_g32", 1.0, [0])
    addA("qkv11",  "requant_g32", 1.0, [0])
    addA("both51", "uniform", 1.0, [0,1,2,3,4])        # faithful magnitude
    addA("plig40", "uniform", 1.0, [0,1,2])            # ablation half
    addA("qkv11",  "uniform", 1.0, [0,1,2])            # ablation half
    addA("both51", "uniform", 0.5, [0,1,2])            # lower bracket

    # ---- Phase 3: method (b) -- routeB served greedy decode vs anchor served decode ----
    @torch.no_grad()
    def eval_b(cfg, mode, alpha, seed, nmax=None):
        names = config_targets(cfg)
        relf = set_config(model, snap, names, mode, alpha=alpha, seed=seed)
        use = seqs if nmax is None else seqs[:nmax]
        prompts_div = 0; first_divs = []
        for s in use:
            pid = s["full"][:s["plen"]]
            _toks, fd = greedy_decode(model, pid, args.cont, ref=s["ref_cont"])
            if fd >= 0:
                prompts_div += 1
                first_divs.append(fd)
        set_config(model, snap, names, "anchor")
        return {
            "phase": "b_decode_vs_decode", "cfg": cfg, "mode": mode, "alpha": alpha, "seed": seed,
            "prompts_diverged": prompts_div, "prompts_total": len(use),
            "first_div_min": (min(first_divs) if first_divs else None),
            "first_div_median": (sorted(first_divs)[len(first_divs)//2] if first_divs else None),
            "rel_frob_mean": relf,
        }

    def addB(cfg, mode, alpha, seed, nmax=None):
        r = eval_b(cfg, mode, alpha, seed, nmax=nmax)
        results.append(r)
        print(f"  B[{cfg:7s} {mode:11s} a={alpha:.2f} s={seed}] served prompts_diverged="
              f"{r['prompts_diverged']}/{r['prompts_total']} first_div_med={r['first_div_median']} "
              f"relF={r['rel_frob_mean']:.4f}", flush=True)

    tb = time.time()
    addB("both51", "anchor", 0.0, 0, nmax=32)          # determinism check (all-or-nothing) -> MUST be 0
    addB("both51", "requant_g32", 1.0, 0)              # concrete realizable served-gate test
    addB("both51", "uniform", 1.0, 0)                  # faithful magnitude served-gate test
    print(f"[phase3] {time.time()-tb:.0f}s", flush=True)

    out = {"results": results, "meta": {
        "num_prompts": len(seqs), "cont": args.cont, "margin_neartie_logit": MARGIN_NEARTIE,
        "plig_modules": len(plig_targets()), "qkv_modules": len(qkv_targets()),
        "both_modules": len(config_targets("both51")),
        "note": "v2 corrected reference (tf-vs-tf + decode-vs-decode); anchor floors must be 0",
    }}
    json.dump(out, open(HERE / "identity_results_v2.json", "w"), indent=2)
    print(f"[done] {time.time()-t0:.0f}s -> identity_results_v2.json", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-prompts", type=int, default=128)
    ap.add_argument("--cont", type=int, default=128)
    ap.add_argument("--max-plen", type=int, default=1024)
    run(ap.parse_args())
