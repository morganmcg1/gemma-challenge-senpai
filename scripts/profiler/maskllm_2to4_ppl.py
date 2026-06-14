#!/usr/bin/env python
"""Phase-1 PPL gate for 2:4 structured sparsity on the verify-GEMM (PR #118).

WHAT THIS MEASURES
------------------
The deployed frontier (`fa2sw_precache_kenyan` -> PLE-folded `osoi5-v0-baked`)
serves an int4 W4A16 (compressed-tensors / Marlin) body at group_size g=128. The
verify-GEMM (core-7 projections: attn q/k/v/o + MLP gate/up/down) is ~53% of the
M=8 decode budget and is weight-bandwidth-bound. 2:4 structured sparsity removes
the streamed bytes of the zeroed weights (~25% weight-byte saving on already-int4
weights: 2 kept int4 + 2-of-4 metadata = 12b vs 16b dense).

The binding numerics gate is **PPL <= 2.42** (program validity, ref 2.30 + 5%).
The greedy-identity gate is SELF-REFERENTIAL (kanna #96 -> #114): a deterministic
pruned checkpoint is token-identical-to-its-OWN-plain-AR by construction, so 2:4
does NOT have to reproduce dense tokens -- it only has to stay under the PPL cap.

OFFLINE FORWARD-PASS PPL (faithful gate)
----------------------------------------
This is a launch-free offline teacher-forced forward pass, NOT a served endpoint.
The checkpoint is decompressed to DENSE bf16 (compressed-tensors run_compressed=
False); the dense weight == dequantized int4. Because each contiguous group-of-4
along the reduction (K) axis shares one g128 scale, magnitude-2:4 on the dense
weight (zero the 2 smallest-|w| per group-of-4) is IDENTICAL to magnitude-2:4 on
the int4 codes -- the survivors stay exactly on the int4 grid and the zeros map to
code 0, so the masked dense weight faithfully represents the served 2:4 int4
checkpoint (no re-quant error on survivors).

The served lm_head is a 16,384-row PCK04 keepset prune (covers all 61,797 scored
GT tokens). We score over that head via the keepset column map, exactly mirroring
the served #52 methodology -- so the offline baseline reproduces served PPL
(measured baseline 2.3812 vs served 2.3772, +0.17%) and the absolute number is
directly comparable to the 2.42 cap. PPL is robust to FP-reduction tie-breaking
(unlike argmax-greedy-identity), so the offline eval is a faithful gate.

RECIPES
-------
* magnitude : keep the 2 largest-|w| per group-of-4 along K (survivors stay on the
              int4 grid -> faithful, no re-quant). Catastrophic at 7B scale.
* sparsegpt : one-shot Hessian/OBS 2:4 (Frantar & Alistarh 2301.00774). Calibrated
              on a held-out chat set (private_proxy, NOT the PPL eval set) -> per-
              layer XtX Hessian -> OBS saliency picks the 2 survivors AND corrects
              them. Survivors go OFF the int4 grid, so we report both the optimistic
              bf16-survivor PPL (a clean lower bound: re-quant only raises it) and
              the faithful re-quantized (int4 g128) PPL. Independent per-layer prune
              (clean upstream activations) for clean per-layer attribution; the
              sequential variant could recover a little, but not the ~+50% gap to
              the +1.6% gate.

MODES
-----
* baseline        : no mask (anchor; must reproduce ~2.38).
* magnitude       : global magnitude-2:4, one PPL pass (`ppl_2to4_magnitude`).
* analyze         : baseline + global + per-layer sweep + greedy safe-subset (mag).
* sparsegpt       : global SparseGPT-2:4, bf16-survivor + re-quant PPL.
* sparsegpt-analyze: baseline + global + per-layer sweep + greedy safe-subset (sgpt).

Group all runs under W&B group `maskllm-2to4-ppl`.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from local_validation import paths  # noqa: E402

DEFAULT_CKPT = "/tmp/osoi5-v0-baked"
DEFAULT_CALIB = "data/private_proxy_sharegpt.json"
CORE7 = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
PPL_GATE = 2.42
SERVED_BASELINE_PPL = 2.3772  # #52 fa2sw_precache_kenyan served


# --------------------------------------------------------------------------- #
# model + keepset
# --------------------------------------------------------------------------- #
def load_model(ckpt: str):
    from transformers import Gemma4ForConditionalGeneration
    from transformers.utils.quantization_config import CompressedTensorsConfig

    t0 = time.time()
    model = Gemma4ForConditionalGeneration.from_pretrained(
        ckpt,
        dtype=torch.bfloat16,
        quantization_config=CompressedTensorsConfig(run_compressed=False),
        low_cpu_mem_usage=True,
        device_map="cuda:0",
    ).eval()
    print(f"[2to4] model loaded in {time.time()-t0:.1f}s "
          f"gpu={torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)
    return model


def build_col_of(ckpt: str):
    ks = json.load(open(Path(ckpt) / "pck04_keepset.json"))
    keep = ks["keep_ids"]
    full = ks["full_vocab"]
    K = ks["pruned_vocab_K"]
    col_of = torch.full((full,), -1, dtype=torch.long)
    col_of[torch.tensor(keep, dtype=torch.long)] = torch.arange(K, dtype=torch.long)
    return col_of.cuda(), K


def load_records():
    ep = paths.import_ppl_endpoint()
    return [ep.normalized_record(r, i) for i, r in enumerate(ep.read_records(paths.ppl_dataset()))]


# --------------------------------------------------------------------------- #
# target discovery
# --------------------------------------------------------------------------- #
def role_of(name: str) -> str | None:
    for r in CORE7:
        if name.endswith(f".{r}"):
            return r
    if name.endswith(".lm_head") or name == "lm_head":
        return "lm_head"
    return None


def layer_idx_of(name: str) -> int | None:
    # ...language_model.layers.<i>.<...>
    parts = name.split(".")
    if "layers" in parts:
        i = parts.index("layers")
        try:
            return int(parts[i + 1])
        except (IndexError, ValueError):
            return None
    return None


def discover_targets(model, roles, include_lmhead: bool, layers):
    """Return list of dicts {name, module, role, layer, in, out, int4_bytes}."""
    targets = []
    for name, mod in model.named_modules():
        w = getattr(mod, "weight", None)
        if w is None or w.dim() != 2:
            continue
        role = role_of(name)
        if role is None:
            continue
        if role == "lm_head":
            if not include_lmhead:
                continue
        else:
            if role not in roles:
                continue
            if ".language_model." not in name:
                continue
        li = layer_idx_of(name)
        if layers is not None and role != "lm_head" and li not in layers:
            continue
        out_f, in_f = w.shape
        targets.append({
            "name": name, "module": mod, "role": role, "layer": li,
            "out": int(out_f), "in": int(in_f),
            "int4_bytes": int(out_f) * int(in_f) // 2,  # 4 bits / weight
        })
    return targets


# --------------------------------------------------------------------------- #
# 2:4 magnitude mask
# --------------------------------------------------------------------------- #
@torch.no_grad()
def magnitude_2to4_(weight: torch.Tensor) -> dict:
    """In-place magnitude-2:4 along the reduction (in/K) axis. Keep 2 largest-|w|
    of each contiguous group-of-4, zero the other 2. Returns nonzero stats."""
    out_f, in_f = weight.shape
    assert in_f % 4 == 0, f"in_features {in_f} not divisible by 4"
    wg = weight.view(out_f, in_f // 4, 4)
    keep_idx = wg.abs().topk(2, dim=-1).indices         # [out, in/4, 2] largest 2
    mask = torch.zeros_like(wg, dtype=torch.bool)
    mask.scatter_(-1, keep_idx, True)
    wg.mul_(mask)
    nz = int(mask.sum().item())
    return {"nonzero": nz, "total": out_f * in_f, "nonzero_frac": nz / (out_f * in_f)}


# --------------------------------------------------------------------------- #
# SparseGPT 2:4  (Frantar & Alistarh, arXiv:2301.00774 -- fasterprune, n:m mode)
# --------------------------------------------------------------------------- #
def load_calib_batches(ckpt: str, calib_path: str, n_seqs: int, maxlen: int):
    """Tokenize the first human turn of each held-out chat conversation through the
    served chat template. Calibration set is DISJOINT from the PPL eval set."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ckpt)
    data = json.load(open(calib_path))

    def _to_ids(obj):
        if isinstance(obj, torch.Tensor):
            return obj if obj.dim() == 2 else obj.unsqueeze(0)
        ids = obj["input_ids"]            # BatchEncoding / dict
        if not isinstance(ids, torch.Tensor):
            ids = torch.tensor(ids, dtype=torch.long)
        return ids if ids.dim() == 2 else ids.unsqueeze(0)

    batches = []
    for ex in data:
        if len(batches) >= n_seqs:
            break
        conv = ex.get("conversations") or []
        text = next((t.get("value") for t in conv if t.get("from") in ("human", "user") and t.get("value")), None)
        if not text:
            continue
        try:
            out = tok.apply_chat_template(
                [{"role": "user", "content": text}], add_generation_prompt=True,
                return_tensors="pt", return_dict=True)
        except Exception:
            out = tok(text, return_tensors="pt")
        ids = _to_ids(out)[:, :maxlen]
        if ids.shape[1] < 8:
            continue
        batches.append(ids.to("cuda:0"))
    return batches


@torch.no_grad()
def capture_hessians(model, group, calib_batches):
    """Accumulate per-linear input second-moment H = (2/N) sum_t x_t x_t^T over the
    calibration set, for the linears in `group` (list of target dicts). Clean
    upstream weights (independent per-layer prune)."""
    H = {t["name"]: torch.zeros(t["in"], t["in"], device="cuda:0", dtype=torch.float32) for t in group}
    cnt = {t["name"]: 0 for t in group}
    handles = []

    def mk(name, in_f):
        def hook(_mod, inp):
            x = inp[0]
            x = x.reshape(-1, in_f).to(torch.float32)
            H[name].addmm_(x.t(), x)
            cnt[name] += x.shape[0]
        return hook

    for t in group:
        handles.append(t["module"].register_forward_pre_hook(mk(t["name"], t["in"])))
    for ids in calib_batches:
        model(input_ids=ids)
    for h in handles:
        h.remove()
    for name in H:
        H[name].mul_(2.0 / max(cnt[name], 1))
    return H, cnt


@torch.no_grad()
def sparsegpt_2to4(W: torch.Tensor, H: torch.Tensor,
                   prune_n: int = 2, prune_m: int = 4,
                   blocksize: int = 128, percdamp: float = 0.01) -> torch.Tensor:
    """One-shot OBS n:m prune (Frantar fasterprune). W is [out, in], H is [in, in].
    Returns the pruned + OBS-corrected fp32 weight (survivors moved off-grid)."""
    W = W.to(torch.float32).clone()
    cols = W.shape[1]
    H = H.clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1.0
    W[:, dead] = 0.0
    damp = percdamp * torch.mean(torch.diag(H))
    di = torch.arange(cols, device=W.device)
    H[di, di] += damp
    H = torch.linalg.cholesky(H)
    H = torch.cholesky_inverse(H)
    H = torch.linalg.cholesky(H, upper=True)
    Hinv = H

    for i1 in range(0, cols, blocksize):
        i2 = min(i1 + blocksize, cols)
        count = i2 - i1
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        mask1 = torch.zeros_like(W1, dtype=torch.bool)
        for i in range(count):
            w = W1[:, i]
            d = Hinv1[i, i]
            if i % prune_m == 0:
                tmp = W1[:, i:i + prune_m] ** 2 / (torch.diag(Hinv1)[i:i + prune_m].reshape(1, -1)) ** 2
                idx = torch.topk(tmp, prune_m - prune_n, dim=1, largest=False)[1]
                mask1.scatter_(1, i + idx, True)
            q = w.clone()
            q[mask1[:, i]] = 0
            Q1[:, i] = q
            err1 = (w - q) / d
            W1[:, i:] -= err1.unsqueeze(1) * Hinv1[i, i:].unsqueeze(0)
            Err1[:, i] = err1
        W[:, i1:i2] = Q1
        W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    return W


@torch.no_grad()
def requant_int4_g128_(W: torch.Tensor, group_size: int = 128) -> torch.Tensor:
    """Symmetric per-(row, in-group) int4 re-quant, matching the served compressed-
    tensors scheme (codes in [-8, 7], scale = max|w_g| / 7). Zeros stay zero."""
    out_f, in_f = W.shape
    g = group_size
    Wg = W.view(out_f, in_f // g, g)
    scale = (Wg.abs().amax(dim=-1, keepdim=True) / 7.0).clamp_min(1e-8)
    q = torch.clamp(torch.round(Wg / scale), -8, 7)
    return (q * scale).reshape(out_f, in_f)


@torch.no_grad()
def build_sparsegpt_cache(model, targets, calib_batches, chunk_layers: int,
                          requant: bool, prune_n: int = 2, prune_m: int = 4):
    """Independent per-layer SparseGPT-2:4 over all targets. Returns:
       cache_bf16   : {name -> pruned bf16 weight on CPU} (survivors at bf16)
       cache_rq     : {name -> pruned+re-quant bf16 weight on CPU} or None
       stats        : per-tensor {name, role, layer, int4_bytes, nonzero_frac, requant_mae}
    Layers are chunked so peak Hessian memory stays bounded (down_proj H is 419 MB)."""
    by_layer = {}
    for t in targets:
        key = "lm_head" if t["role"] == "lm_head" else t["layer"]
        by_layer.setdefault(key, []).append(t)
    keys = sorted(by_layer, key=lambda x: (x == "lm_head", x))

    cache_bf16, cache_rq, stats = {}, ({} if requant else None), []
    t_all = time.time()
    for c0 in range(0, len(keys), chunk_layers):
        chunk_keys = keys[c0:c0 + chunk_layers]
        group = [t for k in chunk_keys for t in by_layer[k]]
        tc = time.time()
        H, cnt = capture_hessians(model, group, calib_batches)
        for t in group:
            W = t["module"].weight
            Wp = sparsegpt_2to4(W, H[t["name"]], prune_n=prune_n, prune_m=prune_m)
            nz_frac = float((Wp != 0).float().mean().item())
            cache_bf16[t["name"]] = Wp.to(torch.bfloat16).cpu()
            rec = {"name": t["name"], "role": t["role"], "layer": t["layer"],
                   "int4_bytes": t["int4_bytes"], "nonzero_frac": nz_frac,
                   "calib_tokens": cnt[t["name"]]}
            if requant:
                Wrq = requant_int4_g128_(Wp)
                rec["requant_mae"] = float((Wrq - Wp).abs().mean().item())
                cache_rq[t["name"]] = Wrq.to(torch.bfloat16).cpu()
            stats.append(rec)
            del Wp
        del H
        torch.cuda.empty_cache()
        print(f"[2to4]   sparsegpt chunk layers {chunk_keys[0]}..{chunk_keys[-1]} "
              f"({len(group)} tensors) in {time.time()-tc:.1f}s "
              f"peak={torch.cuda.max_memory_allocated()/1e9:.2f}GB", flush=True)
    print(f"[2to4] sparsegpt prune done in {time.time()-t_all:.1f}s", flush=True)
    return cache_bf16, cache_rq, stats


# --------------------------------------------------------------------------- #
# PPL
# --------------------------------------------------------------------------- #
@torch.no_grad()
def compute_ppl(model, records, col_of, K, limit=None) -> dict:
    recs = records if limit is None else records[:limit]
    tot_nll = 0.0
    tot_tok = 0
    per_record = []
    for rec in recs:
        ids = torch.tensor(rec["prompt_token_ids"], dtype=torch.long, device="cuda:0").unsqueeze(0)
        ss, se = rec["score_start"], rec["score_end"]
        logits = model(input_ids=ids).logits[0]
        if logits.shape[-1] != K:
            raise RuntimeError(f"logits width {logits.shape[-1]} != keepset {K}")
        lp = torch.log_softmax(logits[ss - 1:se - 1].float(), dim=-1)
        cols = col_of[ids[0, ss:se]]
        if not bool((cols >= 0).all()):
            raise RuntimeError("GT token not in keepset head")
        nll = -lp[torch.arange(se - ss, device="cuda:0"), cols].sum().item()
        tot_nll += nll
        tot_tok += (se - ss)
        per_record.append({"id": rec["id"], "ppl": math.exp(nll / (se - ss)), "ntok": se - ss})
    return {"ppl": math.exp(tot_nll / tot_tok), "nll": tot_nll, "ntok": tot_tok,
            "per_record": per_record}


@torch.no_grad()
def scored_argmax(model, records, col_of, K, limit=None) -> torch.Tensor:
    """Greedy (argmax) token over the scored slice of every record, in keepset-col
    space. Used for the `frac_tokens_flipped_vs_dense` diagnostic (NOT a gate -- the
    greedy-identity gate is self-referential, so flips don't bind; PPL does)."""
    recs = records if limit is None else records[:limit]
    chunks = []
    for rec in recs:
        ids = torch.tensor(rec["prompt_token_ids"], dtype=torch.long, device="cuda:0").unsqueeze(0)
        ss, se = rec["score_start"], rec["score_end"]
        logits = model(input_ids=ids).logits[0]
        chunks.append(logits[ss - 1:se - 1].float().argmax(dim=-1).cpu())
    return torch.cat(chunks)


def flip_fraction(am_dense: torch.Tensor, am_variant: torch.Tensor) -> float:
    return float((am_dense != am_variant).float().mean().item())


# --------------------------------------------------------------------------- #
# mask / restore helpers
# --------------------------------------------------------------------------- #
@torch.no_grad()
def snapshot(targets):
    return {t["name"]: t["module"].weight.detach().to("cpu", copy=True) for t in targets}


@torch.no_grad()
def restore(targets, snap):
    for t in targets:
        t["module"].weight.copy_(snap[t["name"]].to(t["module"].weight.device))


@torch.no_grad()
def apply_magnitude(targets):
    stats = []
    for t in targets:
        s = magnitude_2to4_(t["module"].weight)
        stats.append({**{k: t[k] for k in ("name", "role", "layer", "int4_bytes")}, **s})
    return stats


@torch.no_grad()
def apply_cache(targets, cache):
    for t in targets:
        t["module"].weight.copy_(cache[t["name"]].to(t["module"].weight.device))


# --------------------------------------------------------------------------- #
# per-layer sweep + greedy safe-subset (shared by both recipes)
# --------------------------------------------------------------------------- #
def sweep_and_subset(model, targets, base, records, col_of, K, total_int4_bytes,
                     apply_group, restore_group, limit, sweep_limit):
    """apply_group(grp) prunes a layer-group's live weights; restore_group(grp)
    puts them back. Returns (sweep, cumulative, last_safe)."""
    by_layer = {}
    for t in targets:
        key = "lm_head" if t["role"] == "lm_head" else t["layer"]
        by_layer.setdefault(key, []).append(t)

    sweep = []
    print(f"[2to4] per-layer marginal sweep over {len(by_layer)} groups "
          f"(sweep_limit={sweep_limit}) ...", flush=True)
    t_sweep = time.time()
    for key in sorted(by_layer, key=lambda x: (x == "lm_head", x)):
        grp = by_layer[key]
        apply_group(grp)
        res = compute_ppl(model, records, col_of, K, limit=sweep_limit or limit)
        restore_group(grp)
        gb = sum(t["int4_bytes"] for t in grp)
        sweep.append({"layer": key, "ppl": res["ppl"], "delta": res["ppl"] - base["ppl"],
                      "int4_bytes": gb, "n_tensors": len(grp)})
        print(f"[2to4]   layer {str(key):>7s}: ppl={res['ppl']:.4f} "
              f"delta={res['ppl']-base['ppl']:+.4f}  bytes={gb/1e6:.1f}MB", flush=True)
    print(f"[2to4] sweep done in {time.time()-t_sweep:.1f}s", flush=True)

    ranked = sorted(sweep, key=lambda s: s["delta"])
    cumulative, chosen = [], []
    cum_bytes = 0
    last_safe = None
    over_streak = 0
    print(f"[2to4] greedy cumulative safe-subset (full-set PPL at each step) ...", flush=True)
    for s in ranked:
        key = s["layer"]
        grp = by_layer[key]
        apply_group(grp)
        chosen.append(key)
        cum_bytes += s["int4_bytes"]
        res = compute_ppl(model, records, col_of, K, limit=limit)
        row = {"added_layer": key, "n_layers": len(chosen), "ppl": res["ppl"],
               "delta": res["ppl"] - base["ppl"], "cum_int4_bytes": cum_bytes,
               "cum_byte_frac": cum_bytes / total_int4_bytes, "pass": res["ppl"] <= PPL_GATE}
        cumulative.append(row)
        if res["ppl"] <= PPL_GATE:
            last_safe = row
            over_streak = 0
        else:
            over_streak += 1
        print(f"[2to4]   +{str(key):>7s} ({len(chosen):2d} layers): ppl={res['ppl']:.4f} "
              f"cum_bytes={cum_bytes/1e6:.0f}MB ({100*cum_bytes/total_int4_bytes:.1f}%) "
              f"{'OK' if res['ppl']<=PPL_GATE else 'OVER'}", flush=True)
        if over_streak >= 3 and res["ppl"] > 1.5 * PPL_GATE:
            print(f"[2to4]   (early stop: {over_streak} over-gate adds, ppl>{1.5*PPL_GATE:.2f})", flush=True)
            break
    return sweep, cumulative, last_safe


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--mode", default="analyze",
                    choices=["baseline", "magnitude", "analyze", "sparsegpt", "sparsegpt-analyze", "flipdiag"])
    ap.add_argument("--flip-layers", default=None,
                    help="flipdiag: comma layer subset for the magnitude safe-subset flip-frac")
    ap.add_argument("--roles", default=",".join(CORE7), help="comma role set for the verify-GEMM")
    ap.add_argument("--include-lmhead", action="store_true", help="also 2:4 the lm_head (PPL-sensitive)")
    ap.add_argument("--layers", default=None, help="comma layer subset (default: all)")
    ap.add_argument("--limit", type=int, default=None, help="score first N records (smoke)")
    ap.add_argument("--sweep-limit", type=int, default=None,
                    help="records per PPL pass during the per-layer sweep (default: full)")
    # SparseGPT calibration
    ap.add_argument("--calib", default=DEFAULT_CALIB, help="held-out chat set for Hessian calib")
    ap.add_argument("--calib-seqs", type=int, default=128)
    ap.add_argument("--calib-maxlen", type=int, default=512)
    ap.add_argument("--chunk-layers", type=int, default=6,
                    help="layers per Hessian-capture chunk (bounds peak H memory)")
    ap.add_argument("--no-requant", action="store_true",
                    help="skip int4 re-quant of SparseGPT survivors (report bf16-survivor only)")
    ap.add_argument("--output", default=None)
    ap.add_argument("--wandb_project", default=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"))
    ap.add_argument("--wandb_entity", default=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"))
    ap.add_argument("--wandb_group", default="maskllm-2to4-ppl")
    ap.add_argument("--wandb_name", default="wirbel/maskllm-2to4-ppl")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    for note in paths.prepare_local_gpu_env():
        print(f"[2to4] {note}", flush=True)

    roles = tuple(r.strip() for r in args.roles.split(",") if r.strip())
    layers = None if not args.layers else {int(x) for x in args.layers.split(",")}
    is_sgpt = args.mode in ("sparsegpt", "sparsegpt-analyze")

    model = load_model(args.checkpoint)
    col_of, K = build_col_of(args.checkpoint)
    records = load_records()
    targets = discover_targets(model, roles, args.include_lmhead, layers)
    total_int4_bytes = sum(t["int4_bytes"] for t in targets)
    print(f"[2to4] {len(targets)} target tensors | roles={roles} include_lmhead={args.include_lmhead} "
          f"| int4 weight bytes={total_int4_bytes/1e6:.1f} MB", flush=True)

    payload = {
        "config": {
            "checkpoint": args.checkpoint, "mode": args.mode,
            "recipe": "sparsegpt" if is_sgpt else "magnitude",
            "roles": list(roles), "include_lmhead": args.include_lmhead,
            "layers": sorted(layers) if layers else "all", "ppl_gate": PPL_GATE,
            "served_baseline_ppl": SERVED_BASELINE_PPL, "n_targets": len(targets),
            "target_int4_MB": total_int4_bytes / 1e6, "limit": args.limit,
            "calib": args.calib if is_sgpt else None,
            "calib_seqs": args.calib_seqs if is_sgpt else None,
            "calib_maxlen": args.calib_maxlen if is_sgpt else None,
            "requant": (not args.no_requant) if is_sgpt else None,
            "note": "offline teacher-forced PPL over the 16,384-row PCK04 keepset head; "
                    "dense bf16 == dequantized int4. magnitude-2:4 survivors stay on-grid; "
                    "sparsegpt survivors move off-grid (bf16-survivor = lower bound; re-quant = faithful).",
        },
    }

    # ---- baseline anchor ---------------------------------------------------- #
    t0 = time.time()
    base = compute_ppl(model, records, col_of, K, limit=args.limit)
    print(f"[2to4] BASELINE ppl={base['ppl']:.4f} ntok={base['ntok']} "
          f"({time.time()-t0:.1f}s)  [served #52={SERVED_BASELINE_PPL}]", flush=True)
    payload["baseline_ppl"] = base["ppl"]
    payload["baseline_nll"] = base["nll"]
    payload["baseline_ntok"] = base["ntok"]

    if args.mode == "baseline":
        _finish(args, payload, model)
        return

    snap = snapshot(targets)

    # ======================================================================== #
    # FLIPDIAG: frac_tokens_flipped_vs_dense (secondary diagnostic, NOT a gate)
    # ======================================================================== #
    if args.mode == "flipdiag":
        t0 = time.time()
        am0 = scored_argmax(model, records, col_of, K, limit=args.limit)
        print(f"[2to4] dense greedy tokens captured: {am0.numel()} ({time.time()-t0:.1f}s)", flush=True)
        apply_magnitude(targets)
        amg = scored_argmax(model, records, col_of, K, limit=args.limit)
        restore(targets, snap)
        ff_g = flip_fraction(am0, amg)
        payload["flip_frac_magnitude_global"] = ff_g
        print(f"[2to4] flip_frac magnitude-2:4 GLOBAL = {ff_g:.4f} "
              f"({int(ff_g*am0.numel())}/{am0.numel()} greedy tokens flip)", flush=True)
        if args.flip_layers:
            fl = {int(x) for x in args.flip_layers.split(",")}
            sub = [t for t in targets if t["role"] != "lm_head" and t["layer"] in fl]
            for t in sub:
                magnitude_2to4_(t["module"].weight)
            ams = scored_argmax(model, records, col_of, K, limit=args.limit)
            restore(sub, snap)
            ff_s = flip_fraction(am0, ams)
            payload["flip_frac_magnitude_subset"] = ff_s
            payload["flip_subset_layers"] = sorted(fl)
            print(f"[2to4] flip_frac magnitude-2:4 SUBSET layers={sorted(fl)} = {ff_s:.4f} "
                  f"({int(ff_s*am0.numel())}/{am0.numel()} flip)", flush=True)
        _finish(args, payload, model)
        return

    # ======================================================================== #
    # MAGNITUDE recipe
    # ======================================================================== #
    if not is_sgpt:
        apply_magnitude(targets)
        t0 = time.time()
        glob = compute_ppl(model, records, col_of, K, limit=args.limit)
        print(f"[2to4] GLOBAL magnitude-2:4 (all {len(targets)} targets) "
              f"ppl={glob['ppl']:.4f} delta={glob['ppl']-base['ppl']:+.4f} "
              f"({100*(glob['ppl']-base['ppl'])/base['ppl']:+.2f}%)  "
              f"{'PASS' if glob['ppl']<=PPL_GATE else 'OVER GATE'}  ({time.time()-t0:.1f}s)", flush=True)
        payload["ppl_2to4_magnitude"] = glob["ppl"]
        payload["ppl_2to4_magnitude_delta"] = glob["ppl"] - base["ppl"]
        payload["ppl_2to4_magnitude_pass"] = bool(glob["ppl"] <= PPL_GATE)
        restore(targets, snap)

        if args.mode == "magnitude":
            _finish(args, payload, model)
            return

        def apply_group(grp):
            for t in grp:
                magnitude_2to4_(t["module"].weight)

        def restore_group(grp):
            for t in grp:
                t["module"].weight.copy_(snap[t["name"]].to(t["module"].weight.device))

        sweep, cumulative, last_safe = sweep_and_subset(
            model, targets, base, records, col_of, K, total_int4_bytes,
            apply_group, restore_group, args.limit, args.sweep_limit)
        restore(targets, snap)
        payload["per_layer_sweep"] = sweep
        payload["cumulative_subset"] = cumulative
        payload["safe_subset"] = last_safe
        payload["safe_subset_byte_saving_pct_of_verify_weights"] = (
            100.0 * 0.25 * last_safe["cum_byte_frac"] if last_safe else 0.0)
        _finish(args, payload, model)
        return

    # ======================================================================== #
    # SPARSEGPT recipe
    # ======================================================================== #
    requant = not args.no_requant
    t_cal = time.time()
    calib_batches = load_calib_batches(args.checkpoint, args.calib, args.calib_seqs, args.calib_maxlen)
    calib_tok = sum(b.shape[1] for b in calib_batches)
    print(f"[2to4] calib: {len(calib_batches)} seqs / {calib_tok} tokens from {args.calib} "
          f"(loaded {time.time()-t_cal:.1f}s)", flush=True)
    payload["config"]["calib_n_seqs"] = len(calib_batches)
    payload["config"]["calib_n_tokens"] = calib_tok

    cache_bf16, cache_rq, sgpt_stats = build_sparsegpt_cache(
        model, targets, calib_batches, args.chunk_layers, requant)
    payload["sparsegpt_stats"] = sgpt_stats
    if requant:
        payload["requant_mae_mean"] = sum(s["requant_mae"] for s in sgpt_stats) / len(sgpt_stats)

    # ---- global SparseGPT-2:4 ---------------------------------------------- #
    apply_cache(targets, cache_bf16)
    t0 = time.time()
    g_bf16 = compute_ppl(model, records, col_of, K, limit=args.limit)
    print(f"[2to4] GLOBAL sparsegpt-2:4 bf16-survivor (all {len(targets)}) "
          f"ppl={g_bf16['ppl']:.4f} delta={g_bf16['ppl']-base['ppl']:+.4f} "
          f"({100*(g_bf16['ppl']-base['ppl'])/base['ppl']:+.2f}%)  "
          f"{'PASS' if g_bf16['ppl']<=PPL_GATE else 'OVER GATE'}  ({time.time()-t0:.1f}s)", flush=True)
    restore(targets, snap)
    payload["ppl_2to4_sparsegpt_bf16"] = g_bf16["ppl"]
    payload["ppl_2to4_sparsegpt_bf16_delta"] = g_bf16["ppl"] - base["ppl"]

    ppl_best = g_bf16["ppl"]
    if requant:
        apply_cache(targets, cache_rq)
        t0 = time.time()
        g_rq = compute_ppl(model, records, col_of, K, limit=args.limit)
        print(f"[2to4] GLOBAL sparsegpt-2:4 int4-requant (all {len(targets)}) "
              f"ppl={g_rq['ppl']:.4f} delta={g_rq['ppl']-base['ppl']:+.4f} "
              f"({100*(g_rq['ppl']-base['ppl'])/base['ppl']:+.2f}%)  "
              f"{'PASS' if g_rq['ppl']<=PPL_GATE else 'OVER GATE'}  ({time.time()-t0:.1f}s)", flush=True)
        restore(targets, snap)
        payload["ppl_2to4_sparsegpt_requant"] = g_rq["ppl"]
        payload["ppl_2to4_sparsegpt_requant_delta"] = g_rq["ppl"] - base["ppl"]
        ppl_best = g_rq["ppl"]  # faithful number is the decision-grade gate

    payload["ppl_2to4_sparsegpt"] = ppl_best
    payload["ppl_2to4_sparsegpt_pass"] = bool(ppl_best <= PPL_GATE)

    if args.mode == "sparsegpt":
        _finish(args, payload, model)
        return

    # ---- per-layer sweep + greedy safe-subset (faithful re-quant weights) --- #
    sub_cache = cache_rq if requant else cache_bf16

    def apply_group(grp):
        for t in grp:
            t["module"].weight.copy_(sub_cache[t["name"]].to(t["module"].weight.device))

    def restore_group(grp):
        for t in grp:
            t["module"].weight.copy_(snap[t["name"]].to(t["module"].weight.device))

    sweep, cumulative, last_safe = sweep_and_subset(
        model, targets, base, records, col_of, K, total_int4_bytes,
        apply_group, restore_group, args.limit, args.sweep_limit)
    restore(targets, snap)
    payload["per_layer_sweep"] = sweep
    payload["cumulative_subset"] = cumulative
    payload["safe_subset"] = last_safe
    payload["safe_subset_byte_saving_pct_of_verify_weights"] = (
        100.0 * 0.25 * last_safe["cum_byte_frac"] if last_safe else 0.0)
    _finish(args, payload, model)


def _finish(args, payload, model):
    out = args.output or f"research/maskllm_2to4_ppl/{args.mode}_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[2to4] wrote {out}", flush=True)
    if not args.no_wandb:
        try:
            _log_wandb(args, payload)
        except Exception as exc:  # noqa: BLE001
            print(f"[2to4] W&B logging failed: {exc!r}", flush=True)


def _log_wandb(args, payload):
    import wandb
    run = wandb.init(entity=args.wandb_entity, project=args.wandb_project,
                     group=args.wandb_group, name=args.wandb_name,
                     job_type="profiling", config=payload["config"])
    summary = {k: v for k, v in payload.items()
               if isinstance(v, (int, float, bool, str)) or k in ("baseline_ppl",)}
    if "per_layer_sweep" in payload:
        tbl = wandb.Table(columns=["layer", "ppl", "delta", "int4_MB", "n_tensors"])
        for s in payload["per_layer_sweep"]:
            tbl.add_data(str(s["layer"]), s["ppl"], s["delta"], s["int4_bytes"] / 1e6, s["n_tensors"])
        run.log({"per_layer_sweep": tbl})
    if "cumulative_subset" in payload:
        tbl = wandb.Table(columns=["added_layer", "n_layers", "ppl", "cum_byte_frac", "pass"])
        for s in payload["cumulative_subset"]:
            tbl.add_data(str(s["added_layer"]), s["n_layers"], s["ppl"], s["cum_byte_frac"], s["pass"])
        run.log({"cumulative_subset": tbl})
    run.summary.update(summary)
    run.finish()
    print(f"[2to4] W&B run: {run.url}", flush=True)


if __name__ == "__main__":
    main()
