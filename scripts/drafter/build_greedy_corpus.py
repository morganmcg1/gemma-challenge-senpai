#!/usr/bin/env python3
"""Re-decode the wide corpus prompts along the TARGET's own greedy trajectory.

Why: for *greedy* speculative decoding the only tokens that are ever accepted are
the target's greedy continuation, so the serving-realized (accepted) trajectory IS
the target's greedy decode. v0/v1 teacher-forced the draft on the ShareGPT/MMLU
*reference* text, which is off that trajectory; v1's token-free-run on top of stale
reference-trajectory KL targets collapsed (diverge_frac 0.74, loss exploded). The
fix (HASS-/SeqKD-greedy, arXiv 2408.15766 / 2306.13649) is to teacher-force the
draft on the target's greedy tokens, so the single-prefill KL targets recomputed by
the training loop are COHERENT with the realized token context.

This script only swaps the per-prompt `reference` -> the target's greedy continuation
(stored as exact token ids + decoded text). Same prompts, same dedup, same dist mix
as research/wide_drafter/corpus/train.jsonl -> the ONLY change vs v0 is the training
trajectory. No public-bench prompts are touched (they were already dedup-excluded).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter, defaultdict

import torch

import mtp_common as M


def stratified_order(rows, seed):
    """Interleave rows round-robin across dist so any time-capped prefix stays balanced."""
    import random
    buckets = defaultdict(list)
    for r in rows:
        buckets[r.get("dist", "?")].append(r)
    for d in buckets:
        random.Random(seed).shuffle(buckets[d])
    order, keys = [], list(buckets.keys())
    while any(buckets[k] for k in keys):
        for k in keys:
            if buckets[k]:
                order.append(buckets[k].pop())
    return order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="research/wide_drafter/corpus/train.jsonl")
    ap.add_argument("--out", default="research/wide_drafter/corpus/greedy_train.jsonl")
    ap.add_argument("--target", default="google/gemma-4-E4B-it")
    ap.add_argument("--num-prompts", type=int, default=0, help="0 = all")
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-prompt-tokens", type=int, default=768)
    ap.add_argument("--min-cont-tokens", type=int, default=6,
                    help="drop prompts whose greedy continuation is shorter than this")
    ap.add_argument("--max-minutes", type=float, default=20.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=20260613)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.target)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    eos_ids = set(x for x in [tok.eos_token_id] if x is not None)
    # Gemma uses <end_of_turn> to end an assistant turn; stop there too.
    try:
        eot = tok.convert_tokens_to_ids("<end_of_turn>")
        if isinstance(eot, int) and eot >= 0:
            eos_ids.add(eot)
    except Exception:
        pass

    rows = [json.loads(l) for l in open(args.corpus)]
    rows = stratified_order(rows, args.seed)
    if args.num_prompts:
        rows = rows[:args.num_prompts]
    print(f"[data] {len(rows)} prompts; dist={dict(Counter(r.get('dist') for r in rows))}", flush=True)

    t0 = time.time()
    target = M.load_target(args.target, args.device).eval()
    print(f"[load] target in {time.time()-t0:.1f}s mem={torch.cuda.max_memory_allocated()/1e9:.2f}GB", flush=True)

    # Pre-tokenize chat-formatted prompts once.
    enc_rows = []
    for r in rows:
        ct = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}],
                                     add_generation_prompt=True, return_tensors="pt")
        ids = (ct["input_ids"] if hasattr(ct, "keys") else ct)[0]
        if ids.shape[0] > args.max_prompt_tokens:
            ids = ids[-args.max_prompt_tokens:]
        enc_rows.append((r, ids))
    # Order so a --max-minutes-capped prefix stays DISTRIBUTION-PROPORTIONAL while
    # batches stay ~length-homogeneous. A naive global length-sort (the v0 bug)
    # pushed the long MMLU-Pro/reasoning prompts to the tail, which the time cap
    # then dropped -> reasoning 69 / chat 1731 / math 348 skew, starving the very
    # reasoning slice the wide corpus is supposed to cover. Fix: length-sort WITHIN
    # each dist (short-first, good batching), then interleave by normalized rank so
    # every prefix [0,p] holds ~p*n_d of each dist d (== proportional to the corpus
    # mix). Keeps the chat/ShareGPT private-proxy slice well represented.
    by_dist = defaultdict(list)
    for r, ids in enc_rows:
        by_dist[r.get("dist", "?")].append((r, ids))
    ranked = []
    for d, lst in by_dist.items():
        lst.sort(key=lambda x: x[1].shape[0])
        n = len(lst)
        for rank, item in enumerate(lst):
            ranked.append(((rank + 0.5) / n, item))
    ranked.sort(key=lambda x: x[0])
    enc_rows = [item for _, item in ranked]

    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)
    written = 0
    kept_dist = Counter()
    cont_lens = []
    stopped = False
    with open(args.out, "w") as fh:
        for b0 in range(0, len(enc_rows), args.batch_size):
            if (time.time() - t0) / 60.0 > args.max_minutes:
                print(f"[stop] hit max_minutes={args.max_minutes} after {b0} prompts", flush=True)
                stopped = True
                break
            batch = enc_rows[b0:b0 + args.batch_size]
            maxlen = max(ids.shape[0] for _, ids in batch)
            pad_id = tok.pad_token_id
            input_ids = torch.full((len(batch), maxlen), pad_id, dtype=torch.long)
            attn = torch.zeros((len(batch), maxlen), dtype=torch.long)
            for i, (_, ids) in enumerate(batch):
                input_ids[i, maxlen - ids.shape[0]:] = ids
                attn[i, maxlen - ids.shape[0]:] = 1
            input_ids = input_ids.to(args.device)
            attn = attn.to(args.device)
            with torch.no_grad():
                out = target.generate(input_ids=input_ids, attention_mask=attn,
                                      do_sample=False, max_new_tokens=args.max_new_tokens,
                                      use_cache=True, pad_token_id=pad_id)
            gen = out[:, maxlen:]  # [B, <=max_new_tokens]
            for i, (r, _) in enumerate(batch):
                g = gen[i].tolist()
                # truncate at first eos/end_of_turn
                cut = len(g)
                for j, t in enumerate(g):
                    if t in eos_ids:
                        cut = j
                        break
                g = g[:cut]
                if len(g) < args.min_cont_tokens:
                    continue
                text = tok.decode(g, skip_special_tokens=True)
                rec = {"prompt": r["prompt"], "reference": text, "reference_ids": g,
                       "dist": r.get("dist"), "source": r.get("source"),
                       "prefix_hash": r.get("prefix_hash"), "n_greedy_tokens": len(g)}
                fh.write(json.dumps(rec) + "\n")
                written += 1
                kept_dist[r.get("dist")] += 1
                cont_lens.append(len(g))
            if (b0 // args.batch_size) % 10 == 0:
                el = time.time() - t0
                print(f"[gen] {b0+len(batch)}/{len(enc_rows)} written={written} "
                      f"mean_cont={sum(cont_lens)/max(1,len(cont_lens)):.0f} "
                      f"{el/60:.1f}min mem={torch.cuda.max_memory_allocated()/1e9:.1f}GB", flush=True)

    meta = {"corpus": args.corpus, "out": args.out, "target": args.target,
            "written": written, "requested": len(enc_rows), "stopped_early": stopped,
            "max_new_tokens": args.max_new_tokens, "min_cont_tokens": args.min_cont_tokens,
            "kept_dist": dict(kept_dist),
            "mean_cont_tokens": round(sum(cont_lens) / max(1, len(cont_lens)), 1),
            "minutes": round((time.time() - t0) / 60.0, 2), "seed": args.seed}
    with open(os.path.splitext(args.out)[0] + "_meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    print("=== GREEDY CORPUS META ===", flush=True)
    print(json.dumps(meta, indent=2), flush=True)
    print("BUILD_GREEDY_OK", flush=True)


if __name__ == "__main__":
    main()
