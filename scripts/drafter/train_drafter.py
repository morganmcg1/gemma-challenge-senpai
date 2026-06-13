#!/usr/bin/env python3
"""Teacher-forced, depth-1 KL distillation of the Gemma-4 MTP (EAGLE-style)
drafter on a wide, distribution-matched corpus.

Why teacher-forced depth-1 (and not a standalone-LM trainer like the reference
`train_kl_drafter.py`): the Gemma4 assistant is NOT a standalone LM. Each draft
step conditions on the target's last-layer hidden state + the target's shared KV
for the prefix, with a CONSTANT position id (see scripts/drafter/mtp_common.py
and transformers' SinglePositionMultiTokenCandidateGenerator). So for every
example we:
  1. build seq = chat_template(prompt) + reference  (teacher-forcing text),
  2. run ONE frozen target prefill -> per-position hidden + shared_kv + the
     top-`topk` target next-token distribution,
  3. at sampled continuation positions j, run a depth-1 draft forward
     (token_j, hidden_j | shared_kv[:j+1], pos=j) and minimise
     alpha*CE(target_argmax) + (1-alpha)*KL(target_top-k || draft) over the
     centroid intersection.

Greedy identity is preserved by construction at serve time (spec-decode emits the
target's argmax), so this never touches PPL; it only changes acceptance.

The width hypothesis is tested by `--dist-filter`: train on the full mix (wide)
vs a single distribution like reasoning_mcq (narrow / public-bench-like) and
compare per-distribution held-out acceptance.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import defaultdict

import torch

import mtp_common as M


def load_corpus(path, dist_filter=None, max_examples=None, seed=0):
    rows = []
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            if not r.get("reference"):
                continue
            if dist_filter and r.get("dist") != dist_filter:
                continue
            rows.append(r)
    random.Random(seed).shuffle(rows)
    if max_examples:
        rows = rows[:max_examples]
    return rows


def build_sequence(tok, prompt, reference, max_prompt, max_cont, device):
    """Return (input_ids[1,L], cont_start) with chat-formatted prompt + reference."""
    ct = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                 add_generation_prompt=True, return_tensors="pt")
    chat_ids = (ct["input_ids"] if hasattr(ct, "keys") else ct)
    if chat_ids.shape[1] > max_prompt:
        chat_ids = chat_ids[:, -max_prompt:]
    cont = tok(reference, add_special_tokens=False, return_tensors="pt").input_ids[:, :max_cont]
    if cont.shape[1] < 1:
        return None
    seq = torch.cat([chat_ids, cont], dim=1).to(device)
    return seq, chat_ids.shape[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="research/wide_drafter/corpus/train.jsonl")
    ap.add_argument("--out", default="research/wide_drafter/ckpt/wide")
    ap.add_argument("--target", default="google/gemma-4-E4B-it")
    ap.add_argument("--init", default="google/gemma-4-E4B-it-assistant")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--alpha", type=float, default=0.4, help="CE weight; (1-alpha)=KL weight")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--topk", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--draft-steps", type=int, default=4,
                    help="unrolled teacher-forced draft depth (EAGLE-style; matches serving drift)")
    ap.add_argument("--positions-per-example", type=int, default=10)
    ap.add_argument("--max-prompt-tokens", type=int, default=768)
    ap.add_argument("--max-cont-tokens", type=int, default=256)
    ap.add_argument("--max-examples", type=int, default=0, help="0 = all")
    ap.add_argument("--dist-filter", default="", help="train only on this dist (narrow run)")
    ap.add_argument("--max-minutes", type=float, default=70.0)
    ap.add_argument("--seed", type=int, default=20260613)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--wandb-name", default="")
    ap.add_argument("--wandb-group", default="wide-drafter-distill")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    # Respect the hard senpai bounds.
    env_epochs = os.environ.get("SENPAI_MAX_EPOCHS")
    if env_epochs:
        args.epochs = min(args.epochs, int(env_epochs))
    env_minutes = os.environ.get("SENPAI_TIMEOUT_MINUTES")
    if env_minutes:
        args.max_minutes = min(args.max_minutes, float(env_minutes) - 8.0)  # leave save margin
    if args.debug:
        args.max_examples = args.max_examples or 8
        args.epochs = 1
        args.log_every = 1

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.target)
    t0 = time.time()
    target = M.load_target(args.target, args.device).eval()
    for p in target.parameters():
        p.requires_grad_(False)
    drafter = M.load_drafter(args.init, args.device).train()
    tgt_embed = M.target_input_embeddings(target)
    print(f"[load] in {time.time()-t0:.1f}s mem={torch.cuda.max_memory_allocated()/1e9:.2f}GB", flush=True)

    rows = load_corpus(args.corpus, args.dist_filter or None,
                       args.max_examples or None, args.seed)
    print(f"[data] {len(rows)} examples (dist_filter={args.dist_filter or 'NONE/wide'})", flush=True)

    opt = torch.optim.AdamW([p for p in drafter.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, (len(rows) * args.epochs) // args.grad_accum)

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / max(1, args.warmup)
        import math
        prog = (step - args.warmup) / max(1, total_steps - args.warmup)
        return args.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

    use_wandb = bool(args.wandb_name) and os.environ.get("WANDB_MODE", "online") != "disabled"
    if use_wandb:
        import wandb
        wandb.init(name=args.wandb_name, group=args.wandb_group,
                   config={**vars(args), "total_steps": total_steps,
                           "n_examples": len(rows)})

    rng = random.Random(args.seed)
    gstep = 0          # optimizer steps
    micro = 0          # accumulated examples
    seen_pos = 0
    run_loss = defaultdict(float)
    run_n = 0
    dist_loss = defaultdict(lambda: [0.0, 0])
    opt.zero_grad(set_to_none=True)
    t_train = time.time()
    stop = False

    for epoch in range(args.epochs):
        if stop:
            break
        rng.shuffle(rows)
        for ri, r in enumerate(rows):
            if (time.time() - t_train) / 60.0 > args.max_minutes:
                print(f"[stop] hit max_minutes={args.max_minutes} at epoch{epoch} ex{ri}", flush=True)
                stop = True
                break
            built = build_sequence(tok, r["prompt"], r["reference"],
                                   args.max_prompt_tokens, args.max_cont_tokens, args.device)
            if built is None:
                continue
            seq, cont_start = built
            L = seq.shape[1]
            if L < cont_start + 1 or cont_start < 1:
                continue

            with torch.no_grad():
                last_hidden, shared_kv, logits = M.target_prefill(target, seq)

            # positions j to draft from; need D continuation tokens ahead of j.
            D = args.draft_steps
            lo, hi = cont_start - 1, L - 1 - D
            if hi < lo:
                continue
            cand = list(range(lo, hi + 1))
            rng.shuffle(cand)
            cand = sorted(cand[:args.positions_per_example])

            ex_loss = torch.zeros((), device=args.device)
            kl_acc = ce_acc = fr_acc = 0.0
            nterms = 0
            for j in cand:
                # shared_kv + position id stay FIXED across the K serving draft steps;
                # only (token, hidden) recur. Teacher-force the token, free-run the
                # draft's own hidden (EAGLE) so we train the real serving drift.
                kv_c = M.crop_shared_kv(shared_kv, j + 1)
                cur_tok = seq[:, j:j + 1]
                cur_h = last_hidden[:, j:j + 1, :]
                for d in range(D):
                    d_logits, d_hidden = M.draft_logits_at(drafter, tgt_embed, cur_tok, cur_h,
                                                           kv_c, position_index=j)
                    tgt_row = logits[:, j + d, :].float()          # target dist for token j+d+1
                    tk = torch.topk(tgt_row, k=args.topk, dim=-1)
                    topk_ids = tk.indices
                    topk_probs = torch.softmax(tk.values / args.temperature, dim=-1)
                    argmax_id = tgt_row.argmax(dim=-1)
                    loss, kl, ce, fr = M.kl_ce_loss(d_logits, topk_ids, topk_probs, argmax_id,
                                                    alpha=args.alpha, temperature=args.temperature)
                    ex_loss = ex_loss + loss
                    kl_acc += kl.item(); ce_acc += ce.item(); fr_acc += fr.item(); nterms += 1
                    cur_tok = seq[:, j + d + 1: j + d + 2]          # teacher-forced next token
                    cur_h = d_hidden                                # draft's own hidden
            npos = len(cand)
            ex_loss = ex_loss / max(1, nterms)
            (ex_loss / args.grad_accum).backward()

            run_loss["loss"] += ex_loss.item()
            run_loss["kl"] += kl_acc / max(1, nterms)
            run_loss["ce"] += ce_acc / max(1, nterms)
            run_loss["frac"] += fr_acc / max(1, nterms)
            run_n += 1
            seen_pos += nterms
            dl = dist_loss[r.get("dist", "?")]
            dl[0] += ex_loss.item(); dl[1] += 1
            micro += 1

            if micro % args.grad_accum == 0:
                lr_now = lr_at(gstep)
                for g in opt.param_groups:
                    g["lr"] = lr_now
                gnorm = torch.nn.utils.clip_grad_norm_(drafter.parameters(), args.grad_clip)
                opt.step()
                opt.zero_grad(set_to_none=True)
                gstep += 1
                if gstep % args.log_every == 0:
                    avg = {k: v / max(1, run_n) for k, v in run_loss.items()}
                    toks_s = seen_pos / (time.time() - t_train)
                    msg = (f"[e{epoch} s{gstep}/{total_steps}] loss={avg['loss']:.4f} "
                           f"kl={avg['kl']:.4f} ce={avg['ce']:.4f} frac={avg['frac']:.3f} "
                           f"gn={float(gnorm):.2f} lr={lr_now:.2e} pos/s={toks_s:.0f} "
                           f"mem={torch.cuda.max_memory_allocated()/1e9:.1f}GB")
                    print(msg, flush=True)
                    if use_wandb:
                        import wandb
                        log = {f"train/{k}": v for k, v in avg.items()}
                        log.update({"train/grad_norm": float(gnorm), "train/lr": lr_now,
                                    "train/pos_per_s": toks_s, "step": gstep,
                                    "mem_gb": torch.cuda.max_memory_allocated() / 1e9})
                        for d, (s, n) in dist_loss.items():
                            if n:
                                log[f"train/loss_{d}"] = s / n
                        wandb.log(log)
                    run_loss = defaultdict(float); run_n = 0

    # save
    drafter.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    meta = {"args": vars(args), "n_examples": len(rows), "opt_steps": gstep,
            "positions": seen_pos, "minutes": (time.time() - t_train) / 60.0,
            "dist_loss": {k: (v[0] / v[1] if v[1] else None) for k, v in dist_loss.items()}}
    with open(os.path.join(args.out, "train_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[done] saved -> {args.out}  steps={gstep} pos={seen_pos} "
          f"min={(time.time()-t_train)/60.0:.1f}", flush=True)
    print(json.dumps(meta["dist_loss"], indent=2), flush=True)
    if use_wandb:
        import wandb
        wandb.summary["opt_steps"] = gstep
        wandb.summary["positions"] = seen_pos
        wandb.finish()
    print("TRAIN_OK", flush=True)


if __name__ == "__main__":
    main()
