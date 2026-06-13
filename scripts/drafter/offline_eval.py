#!/usr/bin/env python3
"""Offline acceptance + stability gate for the Gemma-4 MTP drafter.

For a given drafter (stock HF id or a locally trained dir) it measures, per eval
distribution on the held-out shard:
  * mean accepted-tokens/step via NATIVE transformers MTP assisted generation
    (the true serving metric; greedy identity holds so PPL is untouched),
  * greedy-identity pass rate (MTP greedy output == plain greedy output),
  * a teacher-forced per-position acceptance curve P(accept at depth d) along the
    target's own greedy trajectory (reuses the native output, which == greedy).

The per-distribution breakdown is the stability test for the width hypothesis:
a wide-distribution drafter should accept ~uniformly across chat/reasoning/math,
whereas a narrow (public-bench-like) drafter peaks on reasoning and sags on chat.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict

import torch

import mtp_common as M


@torch.no_grad()
def native_accept(target, drafter, tok, ids, max_new_tokens):
    """Run native MTP assisted generation; return (out_ids, new_count, decode_fwds)."""
    calls = {"n": 0}
    lm = getattr(target, "model", target)
    h = lm.register_forward_hook(lambda *a: calls.__setitem__("n", calls["n"] + 1))
    out = target.generate(ids, do_sample=False, max_new_tokens=max_new_tokens,
                          use_cache=True, assistant_model=drafter)
    h.remove()
    new = out[0, ids.shape[1]:]
    return out, new.shape[0], max(1, calls["n"] - 1)


@torch.no_grad()
def plain_greedy(target, ids, max_new_tokens):
    out = target.generate(ids, do_sample=False, max_new_tokens=max_new_tokens, use_cache=True)
    return out[0, ids.shape[1]:]


@torch.no_grad()
def position_curve(target, drafter, tgt_embed, full_ids, prompt_len, K, stride, max_pos):
    """Teacher-forced per-depth acceptance along full_ids (== target greedy).
    Returns (depth_match_counts[K], depth_chain_counts[K], n_positions)."""
    last_hidden, shared_kv, _ = M.target_prefill(target, full_ids)
    L = full_ids.shape[1]
    depth_match = [0] * K     # P(draft[d] == truth[d])  (marginal)
    depth_chain = [0] * K     # P(draft[0..d] all match) (chain)
    npos = 0
    positions = list(range(prompt_len - 1, L - 1 - K, max(1, stride)))[:max_pos]
    for p in positions:
        last_tok = full_ids[:, p:p + 1]
        last_h = last_hidden[:, p:p + 1, :]
        kv_c = M.crop_shared_kv(shared_kv, p + 1)
        drafted, _ = M.propose_k(drafter, tgt_embed, last_tok, last_h, kv_c,
                                 position_index=p, K=K)
        truth = full_ids[0, p + 1:p + 1 + K]
        chain_ok = True
        for d in range(K):
            m = bool(drafted[0, d].item() == truth[d].item())
            if m:
                depth_match[d] += 1
            chain_ok = chain_ok and m
            if chain_ok:
                depth_chain[d] += 1
        npos += 1
    return depth_match, depth_chain, npos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drafter", default="google/gemma-4-E4B-it-assistant",
                    help="HF id (stock) or local trained dir")
    ap.add_argument("--label", default="stock")
    ap.add_argument("--target", default="google/gemma-4-E4B-it")
    ap.add_argument("--heldout", default="research/wide_drafter/corpus/heldout.jsonl")
    ap.add_argument("--max-prompts-per-dist", type=int, default=40)
    ap.add_argument("--new-tokens", type=int, default=96)
    ap.add_argument("--K", type=int, default=7)
    ap.add_argument("--curve-stride", type=int, default=8)
    ap.add_argument("--curve-max-pos", type=int, default=6)
    ap.add_argument("--identity-checks", type=int, default=8, help="prompts to verify greedy identity")
    ap.add_argument("--max-prompt-tokens", type=int, default=768)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="")
    ap.add_argument("--wandb-name", default="")
    ap.add_argument("--wandb-group", default="")
    ap.add_argument("--wandb-run-id", default="",
                    help="resume this W&B run and log the heldout eval into it")
    ap.add_argument("--set-primary", action="store_true",
                    help="also write the advisor primary keys heldout_native/tf_accept_per_step "
                         "(pass for the candidate drafter only)")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.target)
    t0 = time.time()
    target = M.load_target(args.target, args.device).eval()
    drafter = M.load_drafter(args.drafter, args.device).eval()
    if not hasattr(drafter.generation_config, "num_assistant_tokens"):
        pass
    drafter.generation_config.num_assistant_tokens = args.K
    drafter.generation_config.num_assistant_tokens_schedule = "constant"
    tgt_embed = M.target_input_embeddings(target)
    print(f"[load] {args.label} in {time.time()-t0:.1f}s mem={torch.cuda.max_memory_allocated()/1e9:.2f}GB", flush=True)

    # group held-out by dist
    by_dist = defaultdict(list)
    with open(args.heldout) as fh:
        for line in fh:
            r = json.loads(line)
            by_dist[r.get("dist", "?")].append(r)

    results = {}
    overall = {"accept_sum": 0.0, "accept_n": 0, "depth_match": [0] * args.K,
               "depth_chain": [0] * args.K, "npos": 0, "id_ok": 0, "id_n": 0}

    for dist, items in by_dist.items():
        items = items[:args.max_prompts_per_dist]
        acc_sum, acc_n = 0.0, 0
        dmatch = [0] * args.K
        dchain = [0] * args.K
        npos = 0
        id_ok = id_n = 0
        t_d = time.time()
        for idx, r in enumerate(items):
            ct = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}],
                                         add_generation_prompt=True, return_tensors="pt")
            ids = (ct["input_ids"] if hasattr(ct, "keys") else ct)
            if ids.shape[1] > args.max_prompt_tokens:
                ids = ids[:, -args.max_prompt_tokens:]
            ids = ids.to(args.device)

            out, new_n, dfwd = native_accept(target, drafter, tok, ids, args.new_tokens)
            if new_n > 0:
                acc_sum += new_n / dfwd
                acc_n += 1

            if idx < args.identity_checks:
                g = plain_greedy(target, ids, args.new_tokens)
                n = min(g.shape[0], out.shape[1] - ids.shape[1])
                same = bool(torch.equal(g[:n], out[0, ids.shape[1]:ids.shape[1] + n]))
                id_ok += int(same); id_n += 1

            if out.shape[1] - ids.shape[1] >= args.K + 1:
                dm, dc, np_ = position_curve(target, drafter, tgt_embed, out,
                                             ids.shape[1], args.K, args.curve_stride,
                                             args.curve_max_pos)
                for d in range(args.K):
                    dmatch[d] += dm[d]; dchain[d] += dc[d]
                npos += np_

        mean_accept = acc_sum / max(1, acc_n)
        chain_p = [dchain[d] / max(1, npos) for d in range(args.K)]
        # teacher-forced expected accepted tokens/step = 1 bonus + sum_d P(chain accepts d)
        tf_accept = 1.0 + sum(chain_p)
        results[dist] = {
            "n_prompts": len(items),
            "native_accept_per_step": round(mean_accept, 4),
            "tf_accept_per_step": round(tf_accept, 4),
            "depth_chain_p": [round(x, 4) for x in chain_p],
            "depth_match_p": [round(dmatch[d] / max(1, npos), 4) for d in range(args.K)],
            "greedy_identity": f"{id_ok}/{id_n}",
            "npos": npos,
            "sec": round(time.time() - t_d, 1),
        }
        print(f"[{args.label}/{dist}] native_accept={mean_accept:.3f} tf_accept={tf_accept:.3f} "
              f"chain_p={[round(x,3) for x in chain_p]} id={id_ok}/{id_n} "
              f"({len(items)} prompts, {time.time()-t_d:.0f}s)", flush=True)
        overall["accept_sum"] += acc_sum; overall["accept_n"] += acc_n
        for d in range(args.K):
            overall["depth_match"][d] += dmatch[d]; overall["depth_chain"][d] += dchain[d]
        overall["npos"] += npos; overall["id_ok"] += id_ok; overall["id_n"] += id_n

    ov_accept = overall["accept_sum"] / max(1, overall["accept_n"])
    ov_chain = [overall["depth_chain"][d] / max(1, overall["npos"]) for d in range(args.K)]
    summary = {
        "label": args.label,
        "drafter": args.drafter,
        "overall_native_accept_per_step": round(ov_accept, 4),
        "overall_tf_accept_per_step": round(1.0 + sum(ov_chain), 4),
        "overall_depth_chain_p": [round(x, 4) for x in ov_chain],
        "overall_greedy_identity": f"{overall['id_ok']}/{overall['id_n']}",
        "per_dist": results,
        "config": {"new_tokens": args.new_tokens, "K": args.K,
                   "max_prompts_per_dist": args.max_prompts_per_dist},
    }
    print("\n=== SUMMARY ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(summary, fh, indent=2)
        print(f"[saved] {args.out}", flush=True)

    # Log the heldout eval to W&B so the verdict is verifiable from the run, not
    # just the committed JSON. Metrics are namespaced by --label so stock and the
    # candidate can co-exist in the same (resumed) training run.
    wb_on = (bool(args.wandb_name) or bool(args.wandb_run_id)) and \
        os.environ.get("WANDB_MODE", "online") != "disabled"
    if wb_on:
        import wandb
        init_kw = dict(name=args.wandb_name or f"{args.label}-eval",
                       group=args.wandb_group or None)
        if args.wandb_run_id:
            init_kw.update(id=args.wandb_run_id, resume="allow")
        run = wandb.init(**init_kw)
        lab = args.label
        s = run.summary
        s[f"heldout/{lab}/native_accept_per_step"] = summary["overall_native_accept_per_step"]
        s[f"heldout/{lab}/tf_accept_per_step"] = summary["overall_tf_accept_per_step"]
        s[f"heldout/{lab}/greedy_identity"] = summary["overall_greedy_identity"]
        for dist, d in results.items():
            s[f"heldout/{lab}/{dist}/native_accept_per_step"] = d["native_accept_per_step"]
            s[f"heldout/{lab}/{dist}/tf_accept_per_step"] = d["tf_accept_per_step"]
        if args.set_primary:
            s["heldout_native_accept_per_step"] = summary["overall_native_accept_per_step"]
            s["heldout_tf_accept_per_step"] = summary["overall_tf_accept_per_step"]
            # exact key the advisor named (== tf metric measured at this K); keeps the
            # SENPAI-RESULT schema (heldout_accepted_tok_per_step_tf_K7) verifiable from the run.
            s[f"heldout_accepted_tok_per_step_tf_K{args.K}"] = summary["overall_tf_accept_per_step"]
        tbl = wandb.Table(columns=["label", "dist", "depth", "chain_p"])
        for dist, d in results.items():
            for depth, p in enumerate(d["depth_chain_p"], start=1):
                tbl.add_data(lab, dist, depth, p)
        run.log({f"heldout_curve_{lab}": tbl})
        run.finish()
        print(f"[wandb] logged heldout/{lab}/* (run {run.id})", flush=True)
    print("EVAL_OK", flush=True)


if __name__ == "__main__":
    main()
