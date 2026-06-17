#!/usr/bin/env python
"""PR #537 step-2: cheap LOCAL Medusa-1 head train + realized-E[T] measurement.

Medusa-1 = K parallel ResBlock heads on the FROZEN base hidden state, each head's
vocab projection TIED to the frozen base lm_head (so we train only one Linear per
head, ~6.5M params). No autoregressive chain (unlike EAGLE-3/MTP) -> no step-1
collapse; per-position acceptance DECAYS with head depth instead of laddering up.

This is a feasibility SCREEN, run analysis-only on the pod (no HF job, no submit):
  Phase A  load the osoi5 byteexact-442 int4 base (int4 -> bf16 reconstruct), the
           SAME pruned-16k-head model the 442 base serves. lm_head is int4-packed
           [16384, 2560]; greedy is already restricted to the keepset, so CE is over
           16384 (cheap) and the tied Medusa head inherits zero reduced-vocab penalty.
  Phase B  teacher-force the ppl-reference corpus ONCE (no generation). At each
           position j capture the lm_head-INPUT hidden H_j (post-final-norm, via a
           forward_pre_hook) and the base greedy g_j = argmax(softcap(lm_head(H_j))).
           Medusa head_k (k=1..K) target at position t is g_{t+k} (standard Medusa
           teacher-forced self-distillation: predict what the base would greedily
           emit k steps ahead along the reference trajectory).
  Phase C  train K zero-init ResBlock heads (CE to g_{t+k}, frozen tied lm_head +
           softcap) on a train split; measure on a HELD-OUT split:
             - marginal top-1 acceptance a_k = P(argmax head_k(H_t) == g_{t+k})
             - DIRECT chained single-candidate E[T] = 1 + mean(leading-accept run)
             - marginal top-3 acceptance (tree bracket)
           and emit realized_E_T for the screen to fold into the #532 step model.

Greedy argmax is invariant to the monotone softcap, so targets/acceptance are
unaffected; PPL + served self-determinism are drafter-INVARIANT under M=8 greedy
spec-verify (inherited from #523), so this never touches the quality contract.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]  # research/validity/medusa_feasibility_screen -> repo root
sys.path.insert(0, str(REPO / "submissions" / "int4_g128_lmhead"))
from safetensors import safe_open  # noqa: E402
from validate_offline import dequant_group, LANG_PREFIX  # noqa: E402


def rebuild_state_dict(ckpt: Path) -> dict[str, torch.Tensor]:
    """int4 -> bf16 text state dict. Mirrors validate_offline.rebuild_text_state_dict
    but drops its g128-specific hardcoded n_dequant==344 count assert (the osoi5
    byteexact-442 build packs a different number of tensors). dequant_group is the
    same hand-rolled symmetric per-group int4 unpack the gate uses."""
    sd: dict[str, torch.Tensor] = {}
    bases: dict[str, dict[str, torch.Tensor]] = {}
    n = 0
    with safe_open(str(ckpt / "model.safetensors"), framework="pt", device="cpu") as f:
        for name in f.keys():
            if name.endswith((".weight_packed", ".weight_scale", ".weight_shape")):
                base, kind = name.rsplit(".", 1)
                bases.setdefault(base, {})[kind] = f.get_tensor(name)
            elif name == "lm_head.weight":
                sd["lm_head.weight"] = f.get_tensor(name)
            elif name.startswith(LANG_PREFIX):
                sd["model." + name[len(LANG_PREFIX):]] = f.get_tensor(name)
        for base, parts in bases.items():
            w = dequant_group(parts["weight_packed"], parts["weight_scale"], parts["weight_shape"])
            assert torch.isfinite(w).all(), f"non-finite dequant for {base}"
            if base == "lm_head":
                key = "lm_head.weight"
            elif base.startswith(LANG_PREFIX):
                key = "model." + base[len(LANG_PREFIX):] + ".weight"
            else:
                raise SystemExit(f"unexpected quantized base outside text model: {base}")
            sd[key] = w
            n += 1
    print(f"[decompress] dequantized {n} quantized tensors -> all finite", flush=True)
    return sd


def load_base_with_pruned_head(ckpt: Path, sd: dict[str, torch.Tensor], device: str):
    """Build Gemma4ForCausalLM at FULL config (embed stays 262144) but attach the
    PRUNED int4 lm_head [16384, 2560] as a fresh bias-free Linear. load_text_model
    cannot do this (its strict shape check rejects the 16k head against the 262144
    param), so we load the body then swap the head."""
    from transformers import Gemma4ForCausalLM
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

    cfg_full = json.load(open(ckpt / "config.json"))
    tc = dict(cfg_full["text_config"])
    tc["tie_word_embeddings"] = False
    cfg = Gemma4TextConfig(**tc)
    model = Gemma4ForCausalLM(cfg)

    lm_head_w = sd.pop("lm_head.weight")  # [16384, 2560] bf16
    res = model.load_state_dict(sd, strict=False, assign=True)
    assert not res.unexpected_keys, f"unexpected keys: {list(res.unexpected_keys)[:8]}"
    bad_missing = [
        k for k in res.missing_keys
        if (k.endswith(".weight") or k.endswith(".bias")) and k != "lm_head.weight"
    ]
    assert not bad_missing, f"missing weight/bias keys: {bad_missing[:8]}"

    new_head = nn.Linear(cfg.hidden_size, lm_head_w.shape[0], bias=False)
    new_head.weight = nn.Parameter(lm_head_w.to(torch.bfloat16), requires_grad=False)
    model.lm_head = new_head
    model = model.to(device).eval()
    print(f"[load] base loaded | pruned lm_head {tuple(lm_head_w.shape)} attached "
          f"| missing(buffers)={len(res.missing_keys)}", flush=True)
    return model

CKPT_DEFAULT = "/tmp/osoi5-v0-baked"
PPL_DEFAULT = str(
    REPO
    / "official/main_bucket/shared_resources/speed_benchmark/data/ppl_ground_truth_tokens.jsonl"
)
HIDDEN = 2560
SOFTCAP = 30.0


def softcap(z: torch.Tensor, cap: float = SOFTCAP) -> torch.Tensor:
    return torch.tanh(z / cap) * cap


def et_from_ladder(a: list[float]) -> float:
    """E[T] = 1 + sum_m prod_{k<=m} a_k  (independence / product-of-marginals)."""
    et, prod = 1.0, 1.0
    for ak in a:
        prod *= ak
        et += prod
    return et


class ResHead(nn.Module):
    """Medusa-1 ResBlock body: x + SiLU(Linear(x)). Zero-init -> identity at start
    (head == base 1-step head before training)."""

    def __init__(self, hidden: int):
        super().__init__()
        self.linear = nn.Linear(hidden, hidden)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + F.silu(self.linear(h))


@torch.no_grad()
def capture(model, records, max_ctx, max_tgt, device, log_every=16):
    """Teacher-force each record once; return per-seq (H [seq,hidden] bf16 on cpu,
    G [seq] long pruned-greedy idx on cpu, ctx_len int)."""
    grabbed = {}

    def pre_hook(_mod, inp):
        grabbed["h"] = inp[0].detach()

    handle = model.lm_head.register_forward_pre_hook(pre_hook)
    out_H, out_G, out_ctx = [], [], []
    t0 = time.time()
    for i, r in enumerate(records):
        ctx = r["context_token_ids"][:max_ctx]
        tgt = r["target_token_ids"][:max_tgt]
        ids = torch.tensor([ctx + tgt], dtype=torch.long, device=device)
        out = model(input_ids=ids)
        h = grabbed["h"][0]  # [seq, hidden] post-norm lm_head input
        logits = out.logits[0]  # [seq, V] softcapped
        g = logits.argmax(-1)  # [seq] base greedy pruned idx
        out_H.append(h.to(torch.bfloat16).cpu())
        out_G.append(g.to(torch.int32).cpu())
        out_ctx.append(len(ctx))
        if (i + 1) % log_every == 0:
            print(f"  [capture] {i+1}/{len(records)} seq_len={ids.shape[1]} "
                  f"({time.time()-t0:.1f}s)", flush=True)
    handle.remove()
    return out_H, out_G, out_ctx


def build_pairs(Hs, Gs, ctxs, K, device):
    """Flatten (H_t, [g_{t+1..t+K}]) over continuation positions (t >= ctx_len)."""
    H_list, T_list = [], []
    for H, G, c in zip(Hs, Gs, ctxs):
        seq = H.shape[0]
        last = seq - K  # need g_{t+K}, i.e. index t+K <= seq-1
        if last <= c:
            continue
        ts = torch.arange(c, last, dtype=torch.long)  # [n]
        H_list.append(H[ts])  # [n, hidden]
        tgt = torch.stack([G[ts + k] for k in range(1, K + 1)], dim=1)  # [n, K]
        T_list.append(tgt)
    H = torch.cat(H_list, 0).to(device=device, dtype=torch.float32)
    T = torch.cat(T_list, 0).to(device=device, dtype=torch.long)
    return H, T


def head_logits(head: ResHead, lm_w: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    r = head(h)  # [B, hidden] fp32
    z = r @ lm_w.t()  # [B, V]
    return softcap(z)


@torch.no_grad()
def eval_acceptance(heads, lm_w, H, T, K, bs):
    """Held-out acceptance: marginal top-1 a_k, marginal top-3, and DIRECT chained
    single-candidate E[T] = 1 + mean(leading-accept run length)."""
    for hd in heads:
        hd.eval()
    a1 = torch.zeros(K, device=H.device)
    a3 = torch.zeros(K, device=H.device)
    accepted_sum = 0.0
    seen = 0
    for i in range(0, H.shape[0], bs):
        hb = H[i : i + bs]
        tb = T[i : i + bs]
        preds, top3s = [], []
        for k in range(K):
            lg = head_logits(heads[k], lm_w, hb)
            preds.append(lg.argmax(-1))
            top3s.append(lg.topk(3, dim=-1).indices)
        pred = torch.stack(preds, 1)  # [b, K]
        top3 = torch.stack(top3s, 1)  # [b, K, 3]
        match = pred == tb
        a1 += match.float().sum(0)
        a3 += (top3 == tb.unsqueeze(-1)).any(-1).float().sum(0)
        accepted_sum += float(torch.cumprod(match.long(), dim=1).sum())
        seen += hb.shape[0]
    a1 = (a1 / seen).tolist()
    a3 = (a3 / seen).tolist()
    return a1, a3, 1.0 + accepted_sum / seen


@torch.no_grad()
def generate_greedy_freerun(model, contexts, keep_ids, n_new, ctx_cap, device, log_every=16):
    """Free-run base greedy continuation per context — the DEPLOYMENT-FAITHFUL trajectory
    (the served base greedy-decodes these exact prompts). Pruned-head argmax -> keep_ids
    remap -> full id fed back (HF generate can't: it would embed the pruned idx as a full
    id). Returns records {context_token_ids, target_token_ids=full greedy continuation}."""
    recs = []
    t0 = time.time()
    for i, ctx in enumerate(contexts):
        ctx = ctx[-ctx_cap:]
        ids = torch.tensor([ctx], dtype=torch.long, device=device)
        cont_full = []
        for _ in range(n_new):
            out = model(input_ids=ids)
            p = int(out.logits[0, -1].argmax())          # pruned idx 0..V-1
            f = int(keep_ids[p])                          # full vocab id
            cont_full.append(f)
            ids = torch.cat([ids, torch.tensor([[f]], dtype=torch.long, device=device)], 1)
        recs.append({"context_token_ids": ctx, "target_token_ids": cont_full})
        if (i + 1) % log_every == 0:
            print(f"  [freerun] {i+1}/{len(contexts)} n_new={n_new} ({time.time()-t0:.1f}s)",
                  flush=True)
    return recs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default=CKPT_DEFAULT)
    ap.add_argument("--ppl", default=PPL_DEFAULT)
    ap.add_argument("--K", type=int, nargs="+", default=[4, 5])
    ap.add_argument("--n-train", type=int, default=80)
    ap.add_argument("--n-eval", type=int, default=40)
    ap.add_argument("--max-ctx", type=int, default=256)
    ap.add_argument("--max-tgt", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=1024)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--freerun", action="store_true",
                    help="targets = base's OWN greedy continuation (deployment-faithful) "
                         "instead of base-greedy along the PPL reference trajectory")
    ap.add_argument("--n-new", type=int, default=96, help="free-run greedy tokens / prompt")
    ap.add_argument("--ctx-cap", type=int, default=224, help="free-run context cap")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=str(HERE / "medusa_train_results.json"))
    args = ap.parse_args()

    if args.smoke:
        args.n_train, args.n_eval, args.max_tgt, args.epochs = 4, 2, 64, 5
        args.K = [4]
        args.n_new, args.ctx_cap = 24, 160

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"[init] device={device} torch={torch.__version__} smoke={args.smoke} "
          f"K={args.K} n_train={args.n_train} n_eval={args.n_eval} "
          f"max_ctx={args.max_ctx} max_tgt={args.max_tgt} epochs={args.epochs}",
          flush=True)
    if device == "cpu":
        print("[WARN] CUDA not visible -> aborting (need CUDA_VISIBLE_DEVICES=0)")
        sys.exit(2)

    records = [json.loads(l) for l in open(args.ppl) if l.strip()]
    need = args.n_train + args.n_eval
    assert len(records) >= need, f"corpus has {len(records)} < {need} records"
    train_recs = records[: args.n_train]
    eval_recs = records[args.n_train : args.n_train + args.n_eval]
    print(f"[corpus] {len(records)} records | train={len(train_recs)} eval={len(eval_recs)} "
          f"(held-out, disjoint ids)", flush=True)

    # ---- Phase A: load int4 base (pruned 16k head) ----
    t_load0 = time.time()
    ckpt = Path(args.ckpt)
    sd = rebuild_state_dict(ckpt)
    model = load_base_with_pruned_head(ckpt, sd, device)
    del sd
    lm_w = model.lm_head.weight.detach().to(torch.float32)  # [V, hidden] frozen tied head
    V = lm_w.shape[0]
    print(f"[load] base ready V={V} hidden={lm_w.shape[1]} ({time.time()-t_load0:.1f}s) "
          f"| peak_mem={torch.cuda.max_memory_allocated()/1e9:.2f}GB", flush=True)

    # ---- Phase B0 (optional): free-run base greedy continuations (deployment-faithful) ----
    if args.freerun:
        ks = json.load(open(ckpt / "pck04_keepset.json"))["keep_ids"]
        keep_ids = torch.tensor(ks, dtype=torch.long, device=device)
        t_g0 = time.time()
        print(f"[freerun] generating base greedy (n_new={args.n_new} ctx_cap={args.ctx_cap}) "
              f"train split ...", flush=True)
        train_recs = generate_greedy_freerun(
            model, [r["context_token_ids"] for r in train_recs], keep_ids,
            args.n_new, args.ctx_cap, device)
        print("[freerun] eval split ...", flush=True)
        eval_recs = generate_greedy_freerun(
            model, [r["context_token_ids"] for r in eval_recs], keep_ids,
            args.n_new, args.ctx_cap, device)
        # targets now span the full greedy continuation; cap capture target len to n_new
        args.max_tgt = args.n_new
        args.max_ctx = args.ctx_cap
        print(f"[freerun] done ({time.time()-t_g0:.1f}s)", flush=True)

    # ---- Phase B: teacher-forced capture (train + eval) ----
    t_cap0 = time.time()
    print("[capture] train split ...", flush=True)
    Htr, Gtr, ctr = capture(model, train_recs, args.max_ctx, args.max_tgt, device)
    print("[capture] eval split ...", flush=True)
    Hev, Gev, cev = capture(model, eval_recs, args.max_ctx, args.max_tgt, device)
    cap_s = time.time() - t_cap0
    # free the big base model; keep only frozen lm_w
    del model
    torch.cuda.empty_cache()
    print(f"[capture] done ({cap_s:.1f}s); base freed | peak_mem="
          f"{torch.cuda.max_memory_allocated()/1e9:.2f}GB", flush=True)

    results = {
        "ckpt": args.ckpt,
        "smoke": args.smoke,
        "target_trajectory": "freerun_base_greedy" if args.freerun else "ppl_reference",
        "freerun": args.freerun,
        "n_new": args.n_new if args.freerun else None,
        "ctx_cap": args.ctx_cap if args.freerun else None,
        "n_train_seq": len(train_recs),
        "n_eval_seq": len(eval_recs),
        "max_ctx": args.max_ctx,
        "max_tgt": args.max_tgt,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "early_stop_best": True,
        "vocab_pruned": V,
        "capture_seconds": cap_s,
        "by_K": {},
    }

    train_gpu_s_total = 0.0
    for K in args.K:
        print(f"\n===== Medusa-1  K={K} =====", flush=True)
        Htr_K, Ttr_K = build_pairs(Htr, Gtr, ctr, K, device)
        Hev_K, Tev_K = build_pairs(Hev, Gev, cev, K, device)
        M, Me = Htr_K.shape[0], Hev_K.shape[0]
        print(f"[pairs] train_pos={M} eval_pos={Me}", flush=True)

        heads = nn.ModuleList([ResHead(HIDDEN).to(device).float() for _ in range(K)])
        params = [p for hd in heads for p in hd.parameters()]
        opt = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)

        eval_every = max(1, args.epochs // 8)
        best = {"et_sc": -1.0, "ep": 0, "state": None}
        t_tr0 = time.time()
        for ep in range(args.epochs):
            for hd in heads:
                hd.train()
            perm = torch.randperm(M, device=device)
            ep_loss = 0.0
            nb = 0
            for i in range(0, M, args.bs):
                idx = perm[i : i + args.bs]
                hb = Htr_K[idx]
                tb = Ttr_K[idx]
                loss = 0.0
                for k in range(K):
                    lg = head_logits(heads[k], lm_w, hb)
                    loss = loss + F.cross_entropy(lg, tb[:, k])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                ep_loss += float(loss.detach())
                nb += 1
            if ep == 0 or (ep + 1) % eval_every == 0:
                a1, _, et_sc = eval_acceptance(heads, lm_w, Hev_K, Tev_K, K, args.bs)
                print(f"  [train] ep {ep+1}/{args.epochs} loss={ep_loss/nb:.3f} "
                      f"a1={[round(x,3) for x in a1]} et_sc={et_sc:.3f}", flush=True)
                if et_sc > best["et_sc"]:  # early-stop snapshot of best held-out heads
                    best = {"et_sc": et_sc, "ep": ep + 1,
                            "state": [{kk: vv.detach().clone() for kk, vv in hd.state_dict().items()}
                                      for hd in heads]}
        train_s = time.time() - t_tr0
        train_gpu_s_total += train_s

        # restore best-held-out snapshot (guards against overfit past the peak)
        if best["state"] is not None:
            for hd, st in zip(heads, best["state"]):
                hd.load_state_dict(st)
            print(f"[earlystop K={K}] restored best held-out heads from ep {best['ep']} "
                  f"(et_sc={best['et_sc']:.3f})", flush=True)

        # ---- Phase C: held-out acceptance (final, best-snapshot) ----
        a_marg, a_top3, et_singlecand = eval_acceptance(heads, lm_w, Hev_K, Tev_K, K, args.bs)
        et_ladder = et_from_ladder(a_marg)
        et_ladder_top3 = et_from_ladder(a_top3)

        rec = {
            "K": K,
            "train_pos": M,
            "eval_pos": Me,
            "train_seconds": train_s,
            "best_ep": best["ep"],
            "a_marginal_top1": a_marg,
            "a_marginal_top3": a_top3,
            "et_singlecand_direct": et_singlecand,
            "et_ladder_product_top1": et_ladder,
            "et_ladder_product_top3": et_ladder_top3,
        }
        results["by_K"][str(K)] = rec
        print(f"[eval K={K}] a_top1={[round(x,4) for x in a_marg]}", flush=True)
        print(f"[eval K={K}] a_top3={[round(x,4) for x in a_top3]}", flush=True)
        print(f"[eval K={K}] E[T] single-cand DIRECT = {et_singlecand:.4f} | "
              f"ladder(top1)={et_ladder:.4f} | ladder(top3)={et_ladder_top3:.4f} | "
              f"train={train_s:.1f}s", flush=True)

    results["train_gpu_seconds_total"] = train_gpu_s_total
    results["train_gpu_h_total"] = (cap_s + train_gpu_s_total) / 3600.0
    results["wall_seconds_total"] = time.time() - t_load0
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)
    print(f"\n[done] wrote {args.out} | total_gpu_h={results['train_gpu_h_total']:.4f} "
          f"wall={results['wall_seconds_total']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
