# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Offline acceptance eval for the EAGLE-3 draft head (PR #16, Step 4).

Loads a trained checkpoint (`config.json` + `model_*.pt` from train_eagle3.py) and
scores teacher-forced top-1 acceptance on a held-out corpus:

    tf_acceptance_rate = mean( argmax(head_logits[j]) == next_token_ids[j] )

over all non-IGNORE positions, using the same feature/embedding alignment
(`--feature_shift`, default from the checkpoint config) as serving. This is the
single-step, teacher-forced proxy for vLLM draft acceptance (arch_notes S5/S6).

Interpretation bands (PR #16): <0.50 underfit/broken, 0.50-0.70 expected,
>=0.70 strong.

Run (from target/):
  HF_HOME=/senpai-run/home/student-fern/.cache/huggingface \
  python scripts/drafter/eval_eagle3.py \
      --checkpoint research/eagle3_drafter/checkpoints/debug_1k_2ep/ \
      --eval_corpus research/eagle3_drafter/train_data/debug_1k_eval_corpus.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_eagle3 import (  # noqa: E402
    HEAD_DIM,
    Eagle3DraftHead,
    build_rope,
    collate,
    evaluate,
    load_corpus,
    pack_batches,
)


def band(acc: float) -> str:
    if acc < 0.50:
        return "underfit/broken (<0.50)"
    if acc < 0.70:
        return "expected (0.50-0.70)"
    return "strong (>=0.70)"


@torch.no_grad()
def evaluate_topk(head, records, shift, batch_tokens, device, dtype, rope_theta,
                  top_k=4, chunk=256, record_conf=False, conf_top_k=64):
    """Teacher-forced top-1..top-K acceptance + per-position hit-rank traces.

    Extends `train_eagle3.evaluate` (which scores only top-1) to the draft top-K
    candidate set, the quantity a width-K *tree* verifier accepts. For each
    non-IGNORE position j it records `hit_rank` = the 1-indexed rank of the
    reference token `next_token_ids[j]` in the draft's sorted logits if it lands
    in the top-K, else 0 (a top-K miss). From that single tensor:

      topk_acc  = mean(1 <= hit_rank <= k)          (k = 1..K)
      rescue_rate = (topK_acc - top1_acc) / (1 - top1_acc)   (fraction of top-1
                    misses rescued by widening to top-K — fableous's lever)

    The per-record `hit_rank` vectors are returned in sequence order so the
    tree_acceptance_model can simulate the empirical spec-decode accept protocol
    (linear top-1 vs width-K top-K) WITHOUT the i.i.d. assumption. top-1 here is
    bit-identical to `evaluate`'s tf_acceptance_rate (argmax == ref), so it
    self-calibrates against the PR #16 figure.

    `record_conf` (PR #54): additionally capture, per position, the drafter's
    DRAFT-TIME CONFIDENCE statistics over its own emitted next-token distribution
    -- the signals AdaEDL (arXiv:2410.18351) keys early-draft-stop on:
      entropy   = full-vocab Shannon entropy H = -sum p log p   (nats)
      entropy64 = entropy over the renormalised top-`conf_top_k` mass; this is the
                  ONLY entropy a centroid-sparse served head (CENTROID_TOP_K=64,
                  #48) can actually compute, so it is the realistic controller
                  input -- entropy64 <= entropy always (truncation underestimates).
      top1p     = max softmax probability (the Max-Confidence-SPD signal)
      margin    = p1 - p2 (top-1 minus top-2 probability)
    These are pure reductions of the SAME `lcf` already materialised for hit_rank,
    so they add no extra model passes. All four are aligned 1:1 with `hit_rank`,
    letting the entropy_controller measure corr(signal, acceptance / run-length)
    head-to-head vs the acceptance-history r~0.32 baseline (#51).
    """
    head.eval()
    order = list(range(len(records)))
    K = top_k
    hit_at = [0] * (K + 1)          # hit_at[k] = #positions with ref in top-k
    total = 0
    loss_sum = 0.0
    traces = []                     # per record: {seq, n, hit_rank: [...]}
    for idxs in pack_batches(records, batch_tokens, order):
        input_ids, fused, labels, bias, T = collate(records, idxs, shift, device, dtype)
        cos, sin = build_rope(T, HEAD_DIM, rope_theta, device, dtype)
        with torch.autocast("cuda", dtype=dtype):
            hidden = head.forward_hidden(input_ids, fused, cos, sin, bias)  # [B,T,H]
        for bi, rec_idx in enumerate(idxs):
            L = int(records[rec_idx]["input_ids"].shape[0])
            if L <= shift:
                rec = {"seq": int(rec_idx), "n": 0, "hit_rank": []}
                if record_conf:
                    rec.update(entropy=[], entropy64=[], top1p=[], margin=[])
                traces.append(rec)
                continue
            h = hidden[bi, shift:L, :]      # [n, H] — predictions for positions shift..L-1
            ref = labels[bi, shift:L]       # [n]   == next_token_ids[shift:L]
            n = int(h.shape[0])
            ranks = []
            ent, ent64, t1p, marg = [], [], [], []
            for c0 in range(0, n, chunk):
                hc = h[c0:c0 + chunk]
                rc = ref[c0:c0 + chunk]
                with torch.autocast("cuda", dtype=dtype):
                    lc = head.lm_head(hc)
                lcf = lc.float()
                loss_sum += F.cross_entropy(lcf, rc, reduction="sum").item()
                topk_idx = lcf.topk(K, dim=-1).indices       # [c, K], distinct ids
                eqs = topk_idx == rc[:, None]                # [c, K]
                has = eqs.any(-1)
                rank = eqs.float().argmax(-1) + 1            # first match (1-indexed)
                rank = torch.where(has, rank, torch.zeros_like(rank))
                ranks.extend(int(x) for x in rank.tolist())
                if record_conf:
                    logp = F.log_softmax(lcf, dim=-1)        # [c, V] numerically stable
                    p = logp.exp()
                    H = -(p * logp).sum(-1)                  # [c] full-vocab entropy (nats)
                    tv = p.topk(2, dim=-1).values            # [c, 2] top-1/2 probs
                    # top-conf_top_k renormalised entropy (sparse-head realistic):
                    tlv = lcf.topk(conf_top_k, dim=-1).values
                    tlp = F.log_softmax(tlv, dim=-1)
                    H64 = -(tlp.exp() * tlp).sum(-1)         # [c]
                    ent.extend(round(float(x), 5) for x in H.tolist())
                    ent64.extend(round(float(x), 5) for x in H64.tolist())
                    t1p.extend(round(float(x), 5) for x in tv[:, 0].tolist())
                    marg.extend(round(float(x), 5) for x in (tv[:, 0] - tv[:, 1]).tolist())
            rt = torch.tensor(ranks)
            for k in range(1, K + 1):
                hit_at[k] += int(((rt >= 1) & (rt <= k)).sum().item())
            total += n
            rec = {"seq": int(rec_idx), "n": n, "hit_rank": ranks}
            if record_conf:
                rec.update(entropy=ent, entropy64=ent64, top1p=t1p, margin=marg)
            traces.append(rec)
    head.train()
    accs = {k: hit_at[k] / max(1, total) for k in range(1, K + 1)}
    top1, topk_ = accs[1], accs[K]
    rescue = (topk_ - top1) / (1.0 - top1) if top1 < 1.0 else 0.0
    return {"top_acc": accs, "rescue_rate": rescue, "loss": loss_sum / max(1, total),
            "n": total, "traces": traces, "top_k": K, "conf_top_k": conf_top_k}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True,
                    help="dir with config.json + model_*.pt, or a .pt file")
    ap.add_argument("--eval_corpus", "--corpus", dest="eval_corpus", required=True,
                    help="held-out corpus .pt (records: input_ids, aux, next_token_ids)")
    ap.add_argument("--weights", default="model_best.pt",
                    help="weights file inside --checkpoint dir (falls back to last)")
    ap.add_argument("--top_k", "--top-k", dest="top_k", type=int, default=4,
                    help="record top-1..top-K acceptance + rescue_rate (tree width)")
    ap.add_argument("--trace_out", "--trace-out", dest="trace_out", default=None,
                    help="write per-position hit-rank JSONL (one record/sequence) for "
                         "the spec-decode acceptance simulation (Step 3)")
    ap.add_argument("--confidence", action="store_true",
                    help="(PR #54) also record per-position drafter draft-time "
                         "confidence (entropy, entropy64, top1p, margin) into the "
                         "trace, for the AdaEDL entropy-keyed dynamic-K analysis")
    ap.add_argument("--conf_top_k", "--conf-top-k", dest="conf_top_k", type=int,
                    default=64, help="top-k for the renormalised sparse-head entropy "
                                     "proxy (default 64 == served CENTROID_TOP_K)")
    ap.add_argument("--feature_shift", type=int, default=None,
                    help="override; default reads checkpoint config")
    ap.add_argument("--batch_tokens", type=int, default=4096)
    ap.add_argument("--rope_theta", type=float, default=None)
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name", default=None)
    ap.add_argument("--wandb_project", "--wandb-project", dest="wandb_project",
                    default=os.environ.get("WANDB_PROJECT", "senpai-v1"))
    ap.add_argument("--wandb_entity", "--wandb-entity", dest="wandb_entity",
                    default=os.environ.get("WANDB_ENTITY"))
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group",
                    default="eagle3-drafter-training")
    args = ap.parse_args()

    device = "cuda"
    dtype = torch.bfloat16

    if os.path.isdir(args.checkpoint):
        cfg_path = os.path.join(args.checkpoint, "config.json")
        cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
        wpath = os.path.join(args.checkpoint, args.weights)
        if not os.path.exists(wpath):
            wpath = os.path.join(args.checkpoint, "model_last.pt")
    else:
        cfg, wpath = {}, args.checkpoint

    shift = args.feature_shift if args.feature_shift is not None else cfg.get("feature_shift", 1)
    rope_theta = args.rope_theta if args.rope_theta is not None else cfg.get("rope_theta", 1e6)
    norm_before_fc = bool(cfg.get("norm_before_fc", True))

    print(f"[eval] checkpoint: {wpath}", flush=True)
    print(f"[eval] feature_shift={shift} rope_theta={rope_theta:g} "
          f"norm_before_fc={norm_before_fc}", flush=True)

    head = Eagle3DraftHead(norm_before_fc=norm_before_fc).to(device)
    state = torch.load(wpath, map_location=device, weights_only=False)
    missing, unexpected = head.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[eval] load: missing={list(missing)} unexpected={list(unexpected)}",
              flush=True)
    # Ensure frozen tables are bf16 (match training).
    head.model.embed_tokens.weight.data = head.model.embed_tokens.weight.data.to(dtype)
    head.lm_head.weight.data = head.lm_head.weight.data.to(dtype)

    records, _ = load_corpus(args.eval_corpus)
    torch.cuda.reset_peak_memory_stats()
    res = evaluate_topk(head, records, shift, args.batch_tokens, device, dtype,
                        rope_theta, top_k=args.top_k, record_conf=args.confidence,
                        conf_top_k=args.conf_top_k)
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    accs = res["top_acc"]
    acc = accs[1]                      # top-1 == evaluate()'s tf_acceptance_rate
    rescue = res["rescue_rate"]

    print("=" * 60, flush=True)
    print(f"[eval] top-1 tf_acceptance_rate = {acc:.4f}  ({band(acc)})", flush=True)
    for k in range(2, args.top_k + 1):
        print(f"[eval] top-{k} acceptance        = {accs[k]:.4f}", flush=True)
    print(f"[eval] rescue_rate (top-1 misses rescued by top-{args.top_k}) = "
          f"{rescue:.4f}", flush=True)
    print(f"[eval] loss = {res['loss']:.4f}   positions scored = {res['n']}", flush=True)
    print(f"[eval] peak GPU memory = {peak_gb:.2f} GB", flush=True)
    print("=" * 60, flush=True)

    if args.trace_out:
        os.makedirs(os.path.dirname(args.trace_out) or ".", exist_ok=True)
        with open(args.trace_out, "w") as tf:
            tf.write(json.dumps({"meta": {
                "checkpoint": wpath, "eval_corpus": args.eval_corpus,
                "feature_shift": shift, "top_k": args.top_k, "n": res["n"],
                "top_acc": {str(k): v for k, v in accs.items()},
                "rescue_rate": rescue,
                "confidence": bool(args.confidence),
                "conf_top_k": args.conf_top_k}}) + "\n")
            for tr in res["traces"]:
                tf.write(json.dumps(tr) + "\n")
        print(f"[eval] wrote per-position trace -> {args.trace_out} "
              f"({len(res['traces'])} sequences)", flush=True)

    out = {"eval/tf_acceptance_rate": acc, "eval/top1_acc": acc,
           "eval/rescue_rate": rescue, "eval/loss": res["loss"],
           "eval/n": res["n"], "eval/feature_shift": shift, "eval/band": band(acc),
           "eval/peak_gpu_gb": peak_gb}
    for k in range(2, args.top_k + 1):
        out[f"eval/top{k}_acc"] = accs[k]
    if args.wandb_name:
        try:
            import wandb

            run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                             group=args.wandb_group, name=args.wandb_name,
                             config={"checkpoint": wpath, "feature_shift": shift,
                                     "rope_theta": rope_theta, "top_k": args.top_k,
                                     "eval_corpus": args.eval_corpus})
            run.log(out)
            run.summary.update(out)
            run.finish()
        except Exception as e:  # noqa: BLE001
            print(f"[eval] wandb disabled ({e!r})", flush=True)
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
