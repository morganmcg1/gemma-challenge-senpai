#!/usr/bin/env python3
"""Bake-damage locus: offline per-layer hidden + logit divergence (PR #539).

Localizes WHERE the osoi5 reasoning-collapse damage accrues by teacher-forcing the
SAME token stream (base-bf16 greedy completions from #528 decode.jsonl) through three
substrates and comparing residual-stream hidden states + pre-softcap logits:

  1. base-bf16  google/gemma-4-E4B-it            42L, full tied 262k head   (clean ref)
  2. base-int4  google/gemma-4-E4B-it-qat-w4a16-ct 42L, int4 g32, tied head  (int4 knob)
  3. osoi5      /tmp/osoi5-v0-baked               37L, int4 g128, baked 16k head (ship)

Removed original layers (osoi5): {2,3,4,36,37} -> 37 local layers.

KNOB DECOMPOSITION (note the confound, reported explicitly):
  (a) int4 perturbation  = base-bf16 vs base-int4   (1:1, 42L)             [PURE int4 g32]
  (b) layerdrop+         = base-int4 vs osoi5        (depth-aligned + final) [drop + g32->g128 + head-bake]
  base-int4 is g32 / tied-262k; osoi5 is g128 / baked-16k. (b) therefore bundles
  layer-removal with a coarser int4 granularity and the head bake. We measure (b) as
  shipped AND, when --ablate5 is set, an identity-skip of {2,3,4,36,37} inside base-int4
  to isolate PURE layer-removal at g32 (Phase-2 cross-check).

Methodology (researcher-agent, grounded): mask massive-activation dims before cosine
(Gemma huge-norm dims compress cosine->1); report cosine(raw+masked) + relative-MSE;
compare PRE-softcap logits (final_logit_softcapping=30 rotates logit vectors); Block
Influence BI(l)=1-cos(h_in,h_out) on base-int4 to test if removed layers were low-import;
concentration via participation-ratio / Gini / normalized-entropy.

Run (assigned GPU; vllm022 venv has transformers+compressed-tensors):
  CUDA_VISIBLE_DEVICES=0 .venvs/vllm022/bin/python \
      research/validity/bake_damage_locus/forward_divergence.py \
      --n-per-task 20 --max-comp 320 \
      --out research/validity/bake_damage_locus/results.npz \
      [--smoke] [--ablate5]
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

HERE = os.path.dirname(os.path.abspath(__file__))
KCG = os.path.join(os.path.dirname(HERE), "keepset_coverage_gap")

BASE_BF16 = "google/gemma-4-E4B-it"
BASE_INT4 = "google/gemma-4-E4B-it-qat-w4a16-ct"
OSOI5_DEFAULT = "/tmp/osoi5-v0-baked"
REMOVED = [2, 3, 4, 36, 37]           # original 0-indexed layers dropped by osoi5
N_LAYERS_BASE = 42
SOFTCAP = 30.0
TASKS = ["mmlu_pro", "gpqa_diamond", "aime2024"]


def surviving_originals():
    """Original layer index for each osoi5 local layer (37)."""
    return [i for i in range(N_LAYERS_BASE) if i not in set(REMOVED)]


def log(msg):
    print(f"[bdl] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Inputs: rebuild the exact prompt token-ids (chat template) + completion stream
# ---------------------------------------------------------------------------
def build_inputs(n_per_task, max_comp, smoke):
    from transformers import AutoTokenizer

    prompts = {}
    with open(os.path.join(KCG, "prompts.jsonl")) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                prompts[(r["task"], str(r["id"]))] = r["prompt_text"]

    decode = {}
    order = []
    with open(os.path.join(KCG, "decode.jsonl")) as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                key = (d["task"], str(d["id"]))
                decode[key] = d
                order.append(key)

    tok = AutoTokenizer.from_pretrained(BASE_BF16)

    # pick first n_per_task per task in decode order (stable)
    sel = []
    per = {t: 0 for t in TASKS}
    cap = 2 if smoke else n_per_task
    for key in order:
        t = key[0]
        if per.get(t, 0) < cap:
            sel.append(key)
            per[t] += 1
    log(f"selected {len(sel)} prompts: {per}")

    items = []
    mm = 0
    for key in sel:
        d = decode[key]
        ptext = prompts[key]
        msgs = [{"role": "user", "content": ptext}]
        enc = tok.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True, return_dict=True
        )
        pids = list(enc["input_ids"])
        if len(pids) != d["prompt_len"]:
            log(f"WARN prompt_len mismatch {key}: rebuilt {len(pids)} vs stored {d['prompt_len']}")
            mm += 1
        comp = list(d["completion_token_ids"])[:max_comp]
        ids = pids + comp
        items.append({
            "task": key[0], "id": key[1],
            "ids": ids, "prompt_len": len(pids), "comp_len": len(comp),
            "finish": d.get("finish_reason"),
        })
    log(f"built {len(items)} teacher-forcing sequences ({mm} prompt_len mismatches); "
        f"avg_len={sum(len(x['ids']) for x in items)/len(items):.0f}")
    return items


# ---------------------------------------------------------------------------
# Model load / forward
# ---------------------------------------------------------------------------
def load_model(model_id):
    import torch
    from transformers import AutoModelForCausalLM
    t0 = time.time()
    m = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True,
    ).to("cuda").eval()
    log(f"loaded {model_id} in {time.time()-t0:.0f}s; "
        f"mem={torch.cuda.memory_allocated()/1e9:.1f}GB")
    return m


def forward_hidden(model, ids, skip_layers=None):
    """Return (hidden_states tuple on GPU, pre-softcap logits over FULL head on GPU).

    skip_layers: optional set of decoder-layer original indices to replace with identity
    (the ablate5 layer-removal counterfactual). Implemented via forward pre-hooks that
    short-circuit the layer to return its input hidden state unchanged.
    """
    import torch
    handles = []
    if skip_layers:
        layers = _decoder_layers(model)
        for li in skip_layers:
            h = layers[li].register_forward_hook(_identity_layer_hook)
            handles.append(h)
    try:
        t = torch.tensor([ids], device="cuda")
        with torch.no_grad():
            out = model(input_ids=t, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states  # tuple len L+1, each [1,T,H]
        # manual PRE-softcap logits via output embedding (tied or separate head)
        lm = model.get_output_embeddings()
        logits = lm(hs[-1].to(next(lm.parameters()).dtype)).float()  # [1,T,V_head]
    finally:
        for h in handles:
            h.remove()
    return hs, logits


def _decoder_layers(model):
    """Locate the decoder ModuleList across possible Gemma4 wrappers."""
    import torch.nn as nn
    cands = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and name.endswith("layers"):
            cands.append((name, mod))
    # prefer the language-model layer stack (longest, text decoder)
    cands.sort(key=lambda x: len(x[1]))
    return cands[-1][1]


def _identity_layer_hook(module, args, output):
    # decoder layer output is hidden_states or (hidden_states, ...); return input hidden
    inp = args[0]
    if isinstance(output, tuple):
        return (inp,) + tuple(output[1:])
    return inp


# ---------------------------------------------------------------------------
# Divergence primitives (per layer, accumulated over positions)
# ---------------------------------------------------------------------------
def cos_relmse(a, b, mask=None):
    """a,b: [T,H] fp32 (a vs reference b). Returns per-position cos (raw),
    cos (masked), relMSE = ||a-b||^2/||b||^2."""
    import torch
    a = a.float(); b = b.float()
    cos_raw = torch.nn.functional.cosine_similarity(a, b, dim=-1)  # [T]
    if mask is not None:
        am = a * mask; bm = b * mask
    else:
        am, bm = a, b
    cos_m = torch.nn.functional.cosine_similarity(am, bm, dim=-1)
    relmse = ((a - b) ** 2).sum(-1) / (b ** 2).sum(-1).clamp_min(1e-12)
    return cos_raw, cos_m, relmse


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-task", type=int, default=20)
    ap.add_argument("--max-comp", type=int, default=320)
    ap.add_argument("--osoi5-path", default=OSOI5_DEFAULT)
    ap.add_argument("--keepset", default=os.path.join(KCG, "osoi5_baked_keepset_16k.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "results.npz"))
    ap.add_argument("--ablate5", action="store_true",
                    help="also run base-int4 with {2,3,4,36,37} identity-skipped (pure drop)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    import numpy as np
    import torch

    keep_ids = list(json.load(open(args.keepset))["keep_ids"])
    keep_t = torch.tensor(keep_ids, device="cuda")
    n_keep = len(keep_ids)
    log(f"keepset {n_keep} ids")

    items = build_inputs(args.n_per_task, args.max_comp, args.smoke)
    surv = surviving_originals()  # len 37; osoi5 local k -> original surv[k]

    # ---- PASS A: base-int4 (pivot). Cache per-layer hidden (CPU bf16) + keepset logits.
    log("=== PASS A: base-int4 (pivot) ===")
    m = load_model(BASE_INT4)
    torch.cuda.reset_peak_memory_stats()
    int4_hidden = []   # per item: list of [T,H] cpu bf16 (43 layers)
    int4_keeplog = []  # per item: [T,n_keep] cpu fp32 (pre-softcap, keepset)
    # massive-activation detector: per-layer sum|h| over positions, and count
    H = None
    dimabs = None      # [L+1, H]
    dimcnt = 0
    # Block Influence accumulators (base-int4): per layer 1-cos(h_in,h_out)
    bi_sum = np.zeros(N_LAYERS_BASE); bi_cnt = 0
    for it in items:
        hs, logits = forward_hidden(m, it["ids"])
        Lp1 = len(hs); H = hs[0].shape[-1]
        if dimabs is None:
            dimabs = torch.zeros(Lp1, H, device="cuda")
        hids = []
        for l in range(Lp1):
            h = hs[l][0]  # [T,H]
            dimabs[l] += h.float().abs().sum(0)
            hids.append(h.to("cpu", torch.bfloat16))
        dimcnt += hs[0].shape[1]
        int4_hidden.append(hids)
        # keepset pre-softcap logits
        kl = logits[0][:, keep_t].to("cpu")  # [T,n_keep]
        int4_keeplog.append(kl)
        # Block Influence over completion positions
        pl = it["prompt_len"]
        for l in range(N_LAYERS_BASE):
            hin = hs[l][0].float(); hout = hs[l + 1][0].float()
            c = torch.nn.functional.cosine_similarity(hin, hout, dim=-1)  # [T]
            bi_sum[l] += (1 - c).sum().item()
        bi_cnt += hs[0].shape[1]
        del hs, logits
    peakA = torch.cuda.max_memory_allocated() / 1e9
    # massive-activation mask per layer: dims with mean|h| > 100x the 99th pct of that layer
    meanabs = (dimabs / max(dimcnt, 1)).cpu()  # [L+1,H]
    masks = torch.ones_like(meanabs)
    massive_dims = {}
    max_ratio = []
    for l in range(meanabs.shape[0]):
        v = meanabs[l]
        med = float(v.median().clamp_min(1e-9))
        max_ratio.append(float(v.max()) / med)
        # massive = dims whose mean|h| is >=30x the layer's median dim magnitude,
        # capped at 64/layer (Gemma huge-norm dims that compress cosine toward 1).
        big = (v > 30.0 * med).nonzero().flatten().tolist()
        big = sorted(big, key=lambda d: -float(v[d]))[:64]
        masks[l, big] = 0.0
        if big:
            massive_dims[l] = big
    nmask = [len(massive_dims.get(l, [])) for l in range(meanabs.shape[0])]
    log(f"PASS A done; peak {peakA:.1f}GB; massive dims/layer: "
        f"min={min(nmask)} max={max(nmask)} mean={sum(nmask)/len(nmask):.1f}; "
        f"max(max_dim/median) over layers = {max(max_ratio):.0f}")
    bi = bi_sum / max(bi_cnt, 1)
    log(f"base-int4 Block Influence removed-layers {REMOVED}: "
        f"{[round(float(bi[i]),4) for i in REMOVED]}  "
        f"(stack mean {bi.mean():.4f}, max@L{int(bi.argmax())}={bi.max():.4f})")

    # optional ablate5 within base-int4 (pure layer-removal at g32): compute (int4 vs ablate5)
    abl_div = None
    abl_keeplog = None
    if args.ablate5:
        log("=== PASS A2: base-int4 ablate5 (identity-skip removed layers) ===")
        abl_keeplog = []
        # per-layer online divergence int4(full) vs int4(ablate5), aligned 1:1 (both 43)
        acc = _new_acc(N_LAYERS_BASE + 1)
        for i, it in enumerate(items):
            hs, logits = forward_hidden(m, it["ids"], skip_layers=set(REMOVED))
            for l in range(len(hs)):
                ref = int4_hidden[i][l].to("cuda")  # int4 full as reference
                cr, cm, rm = cos_relmse(hs[l][0], ref, masks[l].to("cuda"))
                _accum(acc, l, cr, cm, rm, it["prompt_len"])
            abl_keeplog.append(logits[0][:, keep_t].to("cpu"))
            del hs, logits
        abl_div = _finalize_acc(acc)
    del m; gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()

    # ---- PASS B: base-bf16. Compute (a) bf16-vs-int4 online (ref = bf16).
    log("=== PASS B: base-bf16 ===")
    m = load_model(BASE_BF16)
    bf16_keeplog = []
    accA = _new_acc(N_LAYERS_BASE + 1)  # int4 vs bf16, per layer (ref bf16)
    for i, it in enumerate(items):
        hs, logits = forward_hidden(m, it["ids"])
        for l in range(len(hs)):
            ref = hs[l][0].float()                      # bf16 reference
            cur = int4_hidden[i][l].to("cuda").float()  # int4
            cr, cm, rm = cos_relmse(cur, ref, masks[l].to("cuda"))
            _accum(accA, l, cr, cm, rm, it["prompt_len"])
        bf16_keeplog.append(logits[0][:, keep_t].to("cpu"))
        del hs, logits
    int4_vs_bf16 = _finalize_acc(accA)
    del m; gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()

    # ---- PASS C: osoi5 (37L). Depth-aligned (b) + final-hidden + keepset logits.
    log("=== PASS C: osoi5 ===")
    m = load_model(args.osoi5_path)
    osoi5_keeplog = []
    # per matched original-depth: osoi5 local k (hs[k+1]) vs int4 original surv[k] (hs[surv[k]+1])
    accC = _new_acc(len(surv) + 2)  # +embed(0) +final
    final_div = {"cos_raw": [], "cos_m": [], "relmse": []}
    osoi5_head_width = None
    for i, it in enumerate(items):
        hs, logits = forward_hidden(m, it["ids"])
        osoi5_head_width = logits.shape[-1]
        # embed (index 0) aligns directly
        ref0 = int4_hidden[i][0].to("cuda").float()
        cr, cm, rm = cos_relmse(hs[0][0], ref0, masks[0].to("cuda"))
        _accum(accC, 0, cr, cm, rm, it["prompt_len"])
        # surviving layers
        for k, orig in enumerate(surv):
            cur = hs[k + 1][0]                         # after osoi5 local layer k
            ref = int4_hidden[i][orig + 1].to("cuda")  # after int4 original layer orig
            cr, cm, rm = cos_relmse(cur, ref, masks[orig + 1].to("cuda"))
            _accum(accC, k + 1, cr, cm, rm, it["prompt_len"])
        # final hidden (post final-norm): osoi5 hs[-1] vs int4 hs[-1]
        curf = hs[-1][0]; reff = int4_hidden[i][-1].to("cuda")
        cr, cm, rm = cos_relmse(curf, reff, masks[-1].to("cuda"))
        pl = it["prompt_len"]
        final_div["cos_raw"].append(float(cr[pl:].mean()) if cr[pl:].numel() else float("nan"))
        final_div["cos_m"].append(float(cm[pl:].mean()) if cm[pl:].numel() else float("nan"))
        final_div["relmse"].append(float(rm[pl:].mean()) if rm[pl:].numel() else float("nan"))
        # osoi5 logits: head already over keep rows (16384). map row->vocab via keep order.
        if osoi5_head_width == n_keep:
            osoi5_keeplog.append(logits[0].to("cpu"))         # [T,n_keep] already keepset
        else:
            osoi5_keeplog.append(logits[0][:, keep_t].to("cpu"))  # full head -> restrict
        del hs, logits
    osoi5_vs_int4 = _finalize_acc(accC)
    del m; gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    log(f"osoi5 head width = {osoi5_head_width} (n_keep={n_keep})")

    # ---- First-divergence attribution (d): on completion positions, keepset argmax
    fd = first_divergence(items, bf16_keeplog, int4_keeplog, osoi5_keeplog,
                          keep_ids, abl_keeplog)

    # ---- save everything for the reduction/verdict step
    out = {
        "removed": np.array(REMOVED),
        "surv": np.array(surv),
        "block_influence_int4": bi,
        "massive_dims_count": np.array([len(massive_dims.get(l, [])) for l in range(N_LAYERS_BASE + 1)]),
        "n_items": len(items),
        "n_keep": n_keep,
        "osoi5_head_width": osoi5_head_width or -1,
        "peakA_gb": peakA,
    }
    _pack(out, "int4_vs_bf16", int4_vs_bf16)
    _pack(out, "osoi5_vs_int4", osoi5_vs_int4)
    if abl_div is not None:
        _pack(out, "ablate5_vs_int4", abl_div)
    for k, v in final_div.items():
        out[f"final_{k}"] = np.array(v)
    for k, v in fd.items():
        out[f"fd_{k}"] = np.array(v) if isinstance(v, list) else v
    np.savez(args.out, **out)
    log(f"saved {args.out}")
    # also dump a small json summary of the raw arrays for quick inspection
    json.dump(
        {"n_items": len(items), "n_keep": n_keep, "osoi5_head_width": osoi5_head_width,
         "removed": REMOVED, "fd_summary": fd.get("summary", {})},
        open(args.out.replace(".npz", "_meta.json"), "w"), indent=1)
    log("DONE")
    return 0


# ---- accumulator helpers (per-layer sums over completion positions) ----
def _new_acc(nl):
    import numpy as np
    return {"cos_raw": np.zeros(nl), "cos_m": np.zeros(nl),
            "relmse": np.zeros(nl), "cnt": np.zeros(nl)}


def _accum(acc, l, cos_raw, cos_m, relmse, prompt_len):
    # completion positions only
    cr = cos_raw[prompt_len:]; cm = cos_m[prompt_len:]; rm = relmse[prompt_len:]
    n = cr.numel()
    if n == 0:
        return
    acc["cos_raw"][l] += float(cr.sum())
    acc["cos_m"][l] += float(cm.sum())
    acc["relmse"][l] += float(rm.sum())
    acc["cnt"][l] += n


def _finalize_acc(acc):
    import numpy as np
    cnt = np.clip(acc["cnt"], 1, None)
    return {"cos_raw": acc["cos_raw"] / cnt, "cos_m": acc["cos_m"] / cnt,
            "relmse": acc["relmse"] / cnt, "cnt": acc["cnt"]}


def _pack(out, prefix, d):
    import numpy as np
    for k, v in d.items():
        out[f"{prefix}_{k}"] = np.asarray(v)


def first_divergence(items, bf16_keeplog, int4_keeplog, osoi5_keeplog, keep_ids, abl_keeplog=None):
    """Completion-position keepset-argmax analysis (bf16 is the reference).

    Two attribution views:
      (1) FIRST-FLIP: at osoi5's first off-base flip, was int4 already flipped there
          ('int4-numeric') or not ('layerdrop+')?
      (2) POSITION-MATCHED (robust): over ALL osoi5-flip positions, fraction where int4
          also flips (int4 numeric sufficient) vs osoi5-only (layerdrop+ added). If
          ablate5 available, fraction where the pure-layerdrop ablate5 also flips
          (layerdrop sufficient) -> isolates removal from the g128/head residual.
    Head sanity: bf16 keepset-argmax -> vocab (keep_ids) == teacher token (base full-vocab
    greedy argmax); should be near 1.0 if logit pipeline + keepset ordering are correct."""
    import numpy as np
    import torch

    keep_arr = np.asarray(keep_ids)
    n = len(items)
    int4_first, osoi5_first, driver = [], [], []
    osoi5_div_rate, int4_div_rate = [], []
    # position-matched tallies (pooled over all completion positions of all items)
    tot_osoi5_flip = 0
    osoi5_and_int4 = 0           # int4 numeric sufficient
    osoi5_and_abl = 0           # pure-layerdrop sufficient
    osoi5_only_vs_int4 = 0       # layerdrop+ added beyond int4
    osoi5_only_vs_both = 0       # neither int4 nor ablate5 -> g128/head residual
    teacher_hits = 0; teacher_tot = 0
    for i, it in enumerate(items):
        pl = it["prompt_len"]
        bl = bf16_keeplog[i][pl:]; il = int4_keeplog[i][pl:]; ol = osoi5_keeplog[i][pl:]
        T = bl.shape[0]
        if T == 0:
            int4_first.append(-1); osoi5_first.append(-1); driver.append("none")
            osoi5_div_rate.append(float("nan")); int4_div_rate.append(float("nan"))
            continue
        ba = bl.argmax(-1).numpy(); ia = il.argmax(-1).numpy(); oa = ol.argmax(-1).numpy()
        al = abl_keeplog[i][pl:].argmax(-1).numpy() if abl_keeplog is not None else None
        int4_flip = ia != ba
        osoi5_flip = oa != ba
        i_first = int(np.flatnonzero(int4_flip)[0]) if int4_flip.any() else -1
        o_first = int(np.flatnonzero(osoi5_flip)[0]) if osoi5_flip.any() else -1
        int4_first.append(i_first); osoi5_first.append(o_first)
        int4_div_rate.append(float(int4_flip.mean())); osoi5_div_rate.append(float(osoi5_flip.mean()))
        driver.append(("int4-numeric" if int4_flip[o_first] else "layerdrop") if o_first >= 0 else "none")
        # position-matched (only where osoi5 flips)
        fp = osoi5_flip
        tot_osoi5_flip += int(fp.sum())
        osoi5_and_int4 += int((fp & int4_flip).sum())
        osoi5_only_vs_int4 += int((fp & ~int4_flip).sum())
        if al is not None:
            abl_flip = al != ba
            osoi5_and_abl += int((fp & abl_flip).sum())
            osoi5_only_vs_both += int((fp & ~int4_flip & ~abl_flip).sum())
        # teacher-token sanity: logits at seq pos (pl+t) predict token (pl+t+1), so bf16
        # keepset-argmax[t] -> vocab should equal teacher token[t+1] (base greedy stream).
        tok_next = np.asarray(it["ids"][pl + 1:pl + T + 1])  # len up to T
        k = min(len(tok_next), T - 1) if T > 0 else 0
        if k > 0:
            teacher_hits += int((keep_arr[ba[:k]] == tok_next[:k]).sum()); teacher_tot += k
    drv = np.array(driver); valid = drv != "none"
    of = np.array(osoi5_first); itf = np.array(int4_first)
    summary = {
        "n": n,
        "head_sanity_bf16_keepArgmax_eq_teacher": float(teacher_hits / max(teacher_tot, 1)),
        "osoi5_diverged_frac": float((of >= 0).mean()),
        "int4_diverged_frac": float((itf >= 0).mean()),
        "osoi5_first_median": float(np.median(of[of >= 0])) if (of >= 0).any() else -1.0,
        "int4_first_median": float(np.median(itf[itf >= 0])) if (itf >= 0).any() else -1.0,
        "firstflip_int4_numeric_frac": float((drv[valid] == "int4-numeric").mean()) if valid.any() else float("nan"),
        "firstflip_layerdrop_frac": float((drv[valid] == "layerdrop").mean()) if valid.any() else float("nan"),
        "osoi5_div_rate_mean": float(np.nanmean(osoi5_div_rate)),
        "int4_div_rate_mean": float(np.nanmean(int4_div_rate)),
        # position-matched (pooled)
        "posmatch_total_osoi5_flips": tot_osoi5_flip,
        "posmatch_int4_also_flips_frac": float(osoi5_and_int4 / max(tot_osoi5_flip, 1)),
        "posmatch_osoi5_only_vs_int4_frac": float(osoi5_only_vs_int4 / max(tot_osoi5_flip, 1)),
    }
    if abl_keeplog is not None:
        summary["posmatch_ablate5_also_flips_frac"] = float(osoi5_and_abl / max(tot_osoi5_flip, 1))
        summary["posmatch_residual_g128head_frac"] = float(osoi5_only_vs_both / max(tot_osoi5_flip, 1))
    return {
        "int4_first": int4_first, "osoi5_first": osoi5_first, "driver": list(driver),
        "osoi5_div_rate": osoi5_div_rate, "int4_div_rate": int4_div_rate,
        "summary": summary,
    }


if __name__ == "__main__":
    raise SystemExit(main())
