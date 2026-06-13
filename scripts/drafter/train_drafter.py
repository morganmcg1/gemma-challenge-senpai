#!/usr/bin/env python3
"""Multi-step KL distillation of the Gemma-4 MTP (EAGLE-style) drafter on a
wide, distribution-matched corpus, with a configurable train-time unroll
schedule (v0 HASS-style hidden free-run  ->  v1 EAGLE-3-style token free-run).

Why a custom loop (and not a standalone-LM trainer like the reference
`train_kl_drafter.py`): the Gemma4 assistant is NOT a standalone LM. Each draft
step conditions on the target's last-layer hidden state + the target's shared KV
for the prefix, with a CONSTANT position id (see scripts/drafter/mtp_common.py
and transformers' SinglePositionMultiTokenCandidateGenerator). So for every
example we:
  1. build seq = chat_template(prompt) + reference  (teacher-forcing text),
  2. run ONE frozen target prefill -> per-position hidden + shared_kv + the
     top-`topk` target next-token distribution (the "Option B" on-trajectory
     teacher, reused at every unroll step; no per-step target re-run),
  3. at sampled continuation positions j, unroll the draft `--draft-steps` (K)
     steps from (token_j, hidden_j | shared_kv[:j+1], pos=j) and minimise
     alpha*CE(target_argmax) + (1-alpha)*KL(target_top-k || draft) at each step.

Train-time unroll schedule (`--free-run-prob`, on the greedy-trajectory corpus):
  * Corpus trajectory: build_greedy_corpus.py re-decodes each prompt along the
    TARGET's own greedy continuation (reference_ids). For greedy spec-decode the
    accepted path IS the target greedy decode, so this teacher-forces the draft on
    the serving-realized trajectory (SeqKD/HASS-greedy). v0 teacher-forced on the
    ShareGPT/MMLU *reference* text -- off the greedy path -- which is the v0
    regression (heldout native 3.39 < stock 3.55).
  * The HIDDEN state is ALWAYS free-run across steps (cur_h = draft's own) -- the
    HASS recipe (arXiv 2408.15766), the v0 setting, kept.
  * The TOKEN fed into step d+1 is, with probability `free_run_prob`, the draft's
    OWN argmax (EAGLE-3, arXiv 2503.01840); else the greedy token (teacher-forced).
    The fed-back token is an argmax index -> embedding lookup, naturally detached;
    gradient flows only through the hidden chain (BPTT over K), no straight-through.
  * Teacher is the frozen target softmax at logits[:, j+d] (Option B). This is
    EXACT while the unroll stays on the greedy trajectory. When a free-run step's
    argmax DIVERGES from the greedy token we STOP the chain (rejection-aware break):
    at greedy serve that step is the first mismatch, so every deeper step is
    rejected and must not be supervised against the now-stale on-trajectory target.
    Without the break, free-run tokens + stale Option-B targets BPTT garbage
    gradients into the earlier steps -> the v1 collapse (diverge_frac 0.74,
    chat/math loss 31.9/21.0, native_accept 1.49). `free_run_prob=0.0` is pure
    HASS-greedy (every step teacher-forced on the greedy path).

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


def build_sequence(tok, prompt, reference, max_prompt, max_cont, device, reference_ids=None):
    """Return (input_ids[1,L], cont_start) with chat-formatted prompt + continuation.

    If `reference_ids` is given (the target's exact greedy continuation ids from
    build_greedy_corpus.py), they are used verbatim so the trained trajectory is
    exactly the serving-realized greedy path; otherwise the `reference` text is
    tokenized (the v0 ShareGPT/MMLU reference-text trajectory).
    """
    ct = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                 add_generation_prompt=True, return_tensors="pt")
    chat_ids = (ct["input_ids"] if hasattr(ct, "keys") else ct)
    if chat_ids.shape[1] > max_prompt:
        chat_ids = chat_ids[:, -max_prompt:]
    if reference_ids:
        cont = torch.tensor([reference_ids[:max_cont]], dtype=torch.long)
    else:
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
    ap.add_argument("--lr-decay-by-time", action="store_true",
                    help="cosine-decay LR over elapsed/max_minutes (the ACTUAL budget) instead of "
                         "step/total_steps. Fixes the v0 under-anneal: total_steps was set for N "
                         "epochs but only ~0.87 epoch ran under the time cap, so the step-based cosine "
                         "never reached its tail (LR stuck near 64 pct of peak, losses still falling at "
                         "cutoff). Time-based anneals to ~0 exactly at the budget end -- the EAGLE-3 "
                         "free-running full-budget decay the advisor asked for.")
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--draft-steps", type=int, default=4,
                    help="unrolled draft depth K (matches serving num_assistant_tokens)")
    ap.add_argument("--free-run-prob", type=float, default=0.0,
                    help="prob of feeding the draft's OWN argmax token at unroll steps >=1 "
                         "(EAGLE-3; 0.0 = pure HASS-greedy teacher-forced token). On the "
                         "greedy-trajectory corpus this is made safe by the rejection-aware break: "
                         "a free-run step whose argmax diverges from the greedy token stops the "
                         "chain (never supervised against a stale on-trajectory target). On the v0 "
                         "reference-text trajectory free-run tokens collapsed (diverge_frac 0.74).")
    ap.add_argument("--free-run-ramp-frac", type=float, default=0.1,
                    help="linearly ramp free-run prob 0->target over this fraction of total "
                         "opt steps, then hold (0 = constant free_run_prob from step 0)")
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
    ap.add_argument("--wandb-run-id", default="",
                    help="explicit W&B run id (so offline_eval can resume the same run)")
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

    def lr_at(step, elapsed_min=0.0):
        if step < args.warmup:
            return args.lr * (step + 1) / max(1, args.warmup)
        import math
        if args.lr_decay_by_time:
            prog = elapsed_min / max(1e-6, args.max_minutes)
        else:
            prog = (step - args.warmup) / max(1, total_steps - args.warmup)
        return args.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

    use_wandb = bool(args.wandb_name) and os.environ.get("WANDB_MODE", "online") != "disabled"
    if use_wandb:
        import wandb
        init_kw = dict(name=args.wandb_name, group=args.wandb_group,
                       config={**vars(args), "total_steps": total_steps,
                               "n_examples": len(rows)})
        if args.wandb_run_id:
            init_kw.update(id=args.wandb_run_id, resume="allow")
        run = wandb.init(**init_kw)
        print(f"WANDB_RUN_ID={run.id}", flush=True)

    rng = random.Random(args.seed)
    gstep = 0          # optimizer steps
    micro = 0          # accumulated examples
    seen_pos = 0
    seen_decision = 0  # next-token decisions (K-1 per chain)
    seen_freerun = 0   # decisions where we fed the draft's own token
    seen_diverge = 0   # of those, where it differed from the reference token
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
                                   args.max_prompt_tokens, args.max_cont_tokens, args.device,
                                   reference_ids=r.get("reference_ids"))
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

            # EAGLE-3-style schedule: ramp free-run prob 0 -> target over ramp_frac of training.
            if args.free_run_ramp_frac > 0:
                fr_p = args.free_run_prob * min(1.0, gstep / max(1.0, args.free_run_ramp_frac * total_steps))
            else:
                fr_p = args.free_run_prob

            ex_loss = torch.zeros((), device=args.device)
            div_t = 0  # diverged free-run decisions this example (rejection-aware break)
            kl_acc = ce_acc = fr_acc = 0.0
            nterms = 0
            n_freerun = 0
            for j in cand:
                # shared_kv + position id stay FIXED across the K serving draft steps;
                # only (token, hidden) recur. Free-run the draft's own hidden (HASS) and,
                # with prob fr_p, its own argmax token (EAGLE-3) so we train serving drift.
                kv_c = M.crop_shared_kv(shared_kv, j + 1)
                cur_tok = seq[:, j:j + 1]                  # step-0 input: the verified token (always)
                cur_h = last_hidden[:, j:j + 1, :]         # step-0 input: target's true hidden
                for d in range(D):
                    d_logits, d_hidden = M.draft_logits_at(drafter, tgt_embed, cur_tok, cur_h,
                                                           kv_c, position_index=j)
                    tgt_row = logits[:, j + d, :].float()          # frozen target dist (Option B)
                    tk = torch.topk(tgt_row, k=args.topk, dim=-1)
                    topk_ids = tk.indices
                    topk_probs = torch.softmax(tk.values / args.temperature, dim=-1)
                    argmax_id = tgt_row.argmax(dim=-1)
                    loss, kl, ce, fr = M.kl_ce_loss(d_logits, topk_ids, topk_probs, argmax_id,
                                                    alpha=args.alpha, temperature=args.temperature)
                    ex_loss = ex_loss + loss
                    kl_acc += kl.item(); ce_acc += ce.item(); fr_acc += fr.item(); nterms += 1
                    cur_h = d_hidden                                # draft's own hidden (free-run)
                    if d < D - 1:
                        # next input token: draft's OWN argmax (EAGLE-3, detached index) w.p. fr_p,
                        # else the greedy token (teacher-forced). Same argmax as serve propose_k.
                        true_next = seq[:, j + d + 1: j + d + 2]
                        seen_decision += 1
                        if fr_p > 0.0 and rng.random() < fr_p:
                            draft_next = d_logits.argmax(dim=-1, keepdim=True).detach()
                            n_freerun += 1
                            if bool((draft_next != true_next).item()):
                                # Rejection-aware free-run: on the greedy trajectory this is the
                                # FIRST draft<->target mismatch, so at greedy serve the chain is
                                # rejected here and every deeper step is unreachable. Stop the
                                # unroll instead of feeding the off-trajectory token into the next
                                # step while still scoring it against the on-trajectory (now stale)
                                # target logits[:, j+d+1] -- that stale-target BPTT gradient is the
                                # v1 reference-trajectory collapse (diverge_frac 0.74, loss blow-up).
                                div_t += 1
                                break
                            cur_tok = draft_next  # == true_next: on-trajectory, target coherent
                        else:
                            cur_tok = true_next
            npos = len(cand)
            ex_loss = ex_loss / max(1, nterms)
            (ex_loss / args.grad_accum).backward()

            run_loss["loss"] += ex_loss.item()
            run_loss["kl"] += kl_acc / max(1, nterms)
            run_loss["ce"] += ce_acc / max(1, nterms)
            run_loss["frac"] += fr_acc / max(1, nterms)
            run_n += 1
            seen_pos += nterms
            seen_freerun += n_freerun
            seen_diverge += div_t
            dl = dist_loss[r.get("dist", "?")]
            dl[0] += ex_loss.item(); dl[1] += 1
            micro += 1

            if micro % args.grad_accum == 0:
                lr_now = lr_at(gstep, (time.time() - t_train) / 60.0)
                for g in opt.param_groups:
                    g["lr"] = lr_now
                gnorm = torch.nn.utils.clip_grad_norm_(drafter.parameters(), args.grad_clip)
                opt.step()
                opt.zero_grad(set_to_none=True)
                gstep += 1
                if gstep % args.log_every == 0:
                    avg = {k: v / max(1, run_n) for k, v in run_loss.items()}
                    toks_s = seen_pos / (time.time() - t_train)
                    fr_frac = seen_freerun / max(1, seen_decision)
                    dv_frac = seen_diverge / max(1, seen_freerun)
                    msg = (f"[e{epoch} s{gstep}/{total_steps}] loss={avg['loss']:.4f} "
                           f"kl={avg['kl']:.4f} ce={avg['ce']:.4f} frac={avg['frac']:.3f} "
                           f"frun_p={fr_p:.2f} frun={fr_frac:.2f} div={dv_frac:.2f} "
                           f"gn={float(gnorm):.2f} lr={lr_now:.2e} pos/s={toks_s:.0f} "
                           f"mem={torch.cuda.max_memory_allocated()/1e9:.1f}GB")
                    print(msg, flush=True)
                    if use_wandb:
                        import wandb
                        log = {f"train/{k}": v for k, v in avg.items()}
                        log.update({"train/grad_norm": float(gnorm), "train/lr": lr_now,
                                    "train/pos_per_s": toks_s, "step": gstep,
                                    "train/free_run_prob": fr_p, "train/free_run_frac": fr_frac,
                                    "train/diverge_frac": dv_frac,
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
            "free_run_frac": seen_freerun / max(1, seen_decision),
            "diverge_frac": seen_diverge / max(1, seen_freerun),
            "wandb_run_id": (run.id if use_wandb else None),
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
        wandb.summary["free_run_frac"] = seen_freerun / max(1, seen_decision)
        wandb.summary["diverge_frac"] = seen_diverge / max(1, seen_freerun)
        wandb.finish()
    print("TRAIN_OK", flush=True)


if __name__ == "__main__":
    main()
