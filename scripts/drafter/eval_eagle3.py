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

With `--native`, also runs a free-running EAGLE chain simulation that measures
`native_accept_per_step` — the serving-side accepted-tokens-per-round, where draft
steps 2..K consume the draft's OWN rolled-forward hidden (not the real target
feature). tf is the UPPER BOUND; native is what converts to TPS. `native_step1_top1`
equals the tf top-1 by construction (a wiring self-check). See `evaluate_native`.

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
    HID,
    IGNORE,
    N_AUX,
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
                  top_k=4, chunk=256):
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
                traces.append({"seq": int(rec_idx), "n": 0, "hit_rank": []})
                continue
            h = hidden[bi, shift:L, :]      # [m, H] — predictions for positions shift..L-1
            ref = labels[bi, shift:L]       # [m]   == next_token_ids[shift:L]
            # Skip IGNORE positions so response-only-masked corpora (PR #34) score
            # only the generated continuation; unmasked corpora are unaffected (all
            # positions valid). Keeps top-1 == train_eagle3.evaluate's tf_acc.
            valid = ref != IGNORE
            h = h[valid]
            ref = ref[valid]
            n = int(h.shape[0])
            if n == 0:
                traces.append({"seq": int(rec_idx), "n": 0, "hit_rank": []})
                continue
            ranks = []
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
            rt = torch.tensor(ranks)
            for k in range(1, K + 1):
                hit_at[k] += int(((rt >= 1) & (rt <= k)).sum().item())
            total += n
            traces.append({"seq": int(rec_idx), "n": n, "hit_rank": ranks})
    head.train()
    accs = {k: hit_at[k] / max(1, total) for k in range(1, K + 1)}
    top1, topk_ = accs[1], accs[K]
    rescue = (topk_ - top1) / (1.0 - top1) if top1 < 1.0 else 0.0
    return {"top_acc": accs, "rescue_rate": rescue, "loss": loss_sum / max(1, total),
            "n": total, "traces": traces, "top_k": K}


def _draft_step(head, input_ids, h0, n, rope_theta, device, dtype):
    """One draft-body forward over a length-n sequence with a PRECOMPUTED post-fc
    h0 (bypasses `combine`/`fc`). Replicates `DraftBody.forward` exactly except the
    h0 source, so real-prefix positions match the teacher-forced path bit-for-bit
    while speculative positions can carry the draft's OWN rolled-forward hidden.
    Returns the draft output hidden [n, H] (one decoder layer + final norm)."""
    cos, sin = build_rope(n, HEAD_DIM, rope_theta, device, dtype)
    causal = torch.tril(torch.ones(n, n, dtype=torch.bool, device=device))
    bias = torch.zeros(1, 1, n, n, dtype=dtype, device=device)
    bias.masked_fill_(~causal[None, None], float("-inf"))
    embeds = head.model.embed_tokens(input_ids)
    with torch.autocast("cuda", dtype=dtype):
        mlp_out, res1 = head.model.layers[0](embeds, h0, cos, sin, bias)
        out = head.model.norm(mlp_out + res1)
    return out[0]  # [n, H]


@torch.no_grad()
def evaluate_native(head, records, shift, device, dtype, rope_theta,
                    chain_k=8, max_starts=16):
    """Free-running (native) EAGLE chain acceptance — the serving-side objective.

    Teacher-forced eval feeds the REAL target feature at every position, so it is
    an UPPER BOUND on what serving accepts. At serve time only the FIRST draft step
    of each verification round sees a real target feature; steps 2..K consume the
    draft's OWN output hidden (the post-fc residual fed forward) plus the draft's
    OWN guessed token. That 'interface-fidelity gap' (advisor note on PR #34, from
    PR #9's cross-finding that tf and native acceptance can anti-correlate) is what
    this metric captures, and it is the quantity that converts to TPS.

    For each sampled response start p (a valid verification-round start: every real
    position has a real target feature) we draft a chain of up to `chain_k` tokens:

        step 1: ( h0 = combine(fused[p-shift]) [REAL],   embed(x_p)      ) -> g_{p+1}, d_p
        step k: ( h0 = d_{p+k-1}               [DRAFT-OWN], embed(g_{p+k-1}) ) -> g_{p+k}, d_{p+k-1}

    and accept the leading run of g's that match the target greedy continuation
    (`next_token_ids`). The draft attends over the full real prefix + the accepted
    speculative tail each step (causal self-attention, re-run for simplicity).

      native_accept_per_step = mean accepted run length over starts  (advisor's
                               'accepted-tok/step'; tokens/target-forward = this + 1)
      native_step1_top1      = fraction of starts whose step-1 draft is correct;
                               this MUST equal the tf top-1 (step 1 uses real inputs)
                               and is a built-in wiring self-check.

    Returns per-source means too (mmlu_pro / gpqa / aime) so we can see where the
    benchmark-matched corpus moves the *serving* number, not just the tf bound.
    """
    head.eval()
    run_by_src = {}
    step1_hit = step1_tot = 0
    for rec in records:
        ids_full = rec["input_ids"].to(torch.long)
        L = int(ids_full.shape[0])
        if L < 3:
            continue
        nxt = rec["next_token_ids"]
        src = rec.get("source", "all")
        scored = [a for a in range(L - 1) if int(nxt[a]) != IGNORE]
        if not scored:
            continue
        # Real-prefix post-fc h0 for the whole sequence (matches collate's roll).
        aux = rec["aux"].to(device, torch.float32)                  # [3, L, H]
        fused = aux.permute(1, 0, 2).reshape(L, N_AUX * HID)         # [L, 7680]
        rolled = torch.zeros_like(fused)
        if shift > 0:
            rolled[shift:] = fused[:L - shift]
        else:
            rolled = fused
        with torch.autocast("cuda", dtype=dtype):
            h0_real = head.model.combine(rolled[None].to(dtype))[0]  # [L, H]
        ids_dev = ids_full.to(device)

        if len(scored) > max_starts:
            sel = torch.linspace(0, len(scored) - 1, max_starts).round().long().tolist()
            starts = sorted(set(scored[i] for i in sel))
        else:
            starts = scored

        for p in starts:
            seq_ids = ids_dev[:p + 1].clone()
            seq_h0 = h0_real[:p + 1].clone()
            d_prev = _draft_step(head, seq_ids[None], seq_h0[None], p + 1,
                                 rope_theta, device, dtype)[p]
            with torch.autocast("cuda", dtype=dtype):
                g = int(head.lm_head(d_prev[None]).float().argmax(-1).item())
            step1_tot += 1
            if g == int(ids_full[p + 1]):
                step1_hit += 1
            accepted = 0
            for k in range(chain_k):
                pos = p + 1 + k
                if pos >= L or g != int(ids_full[pos]):
                    break
                accepted += 1
                if pos + 1 >= L:
                    break
                # Roll forward: append accepted token (== target) with DRAFT-OWN h0.
                seq_ids = torch.cat([seq_ids, ids_dev[pos:pos + 1]])
                seq_h0 = torch.cat([seq_h0, d_prev[None]], dim=0)
                d_prev = _draft_step(head, seq_ids[None], seq_h0[None], pos + 1,
                                     rope_theta, device, dtype)[pos]
                with torch.autocast("cuda", dtype=dtype):
                    g = int(head.lm_head(d_prev[None]).float().argmax(-1).item())
            run_by_src.setdefault(src, []).append(accepted)

    head.train()
    all_runs = [r for v in run_by_src.values() for r in v]
    aps = sum(all_runs) / max(1, len(all_runs))
    per_src = {s: sum(v) / max(1, len(v)) for s, v in run_by_src.items()}
    per_src_n = {s: len(v) for s, v in run_by_src.items()}
    return {"native_accept_per_step": aps,
            "native_step1_top1": step1_hit / max(1, step1_tot),
            "per_source": per_src, "per_source_n": per_src_n,
            "n_starts": len(all_runs), "chain_k": chain_k}


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
    ap.add_argument("--native", action="store_true",
                    help="also run the free-running (native) chain-acceptance sim — "
                         "the serving-side accept/step, complementing the tf bound")
    ap.add_argument("--native_k", "--native-k", dest="native_k", type=int, default=8,
                    help="native chain depth K (drafted tokens per verification round)")
    ap.add_argument("--native_starts", "--native-starts", dest="native_starts",
                    type=int, default=16,
                    help="max sampled round-start positions per held-out sequence")
    ap.add_argument("--trace_out", "--trace-out", dest="trace_out", default=None,
                    help="write per-position hit-rank JSONL (one record/sequence) for "
                         "the spec-decode acceptance simulation (Step 3)")
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
                        rope_theta, top_k=args.top_k)
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

    # Per-source breakdown (PR #34): holdout records carry a "source" field
    # (mmlu_pro / gpqa / aime). hit_rank==1 is a top-1 hit; 1<=rank<=K a top-K hit.
    # Aggregating per source shows WHERE the benchmark-matched corpus moves
    # acceptance, the comparison the PR asks for. Legacy corpora (no source) fall
    # back to a single "all" bucket, so this is a no-op there.
    src_metrics = {}
    by_src = {}
    for tr in res["traces"]:
        src = records[tr["seq"]].get("source", "all")
        d = by_src.setdefault(src, {"hit1": 0, "hitk": 0, "n": 0})
        for r in tr["hit_rank"]:
            if r == 1:
                d["hit1"] += 1
            if 1 <= r <= args.top_k:
                d["hitk"] += 1
        d["n"] += tr["n"]
    if len(by_src) > 1 or "all" not in by_src:
        for src in sorted(by_src):
            d = by_src[src]
            s1 = d["hit1"] / max(1, d["n"])
            sk = d["hitk"] / max(1, d["n"])
            print(f"[eval]   source {src:9s}: top1={s1:.4f} top{args.top_k}={sk:.4f} "
                  f"(positions={d['n']})", flush=True)
            src_metrics[f"eval/src_{src}_top1"] = s1
            src_metrics[f"eval/src_{src}_top{args.top_k}"] = sk
            src_metrics[f"eval/src_{src}_n"] = d["n"]
        print("=" * 60, flush=True)

    if args.trace_out:
        os.makedirs(os.path.dirname(args.trace_out) or ".", exist_ok=True)
        with open(args.trace_out, "w") as tf:
            tf.write(json.dumps({"meta": {
                "checkpoint": wpath, "eval_corpus": args.eval_corpus,
                "feature_shift": shift, "top_k": args.top_k, "n": res["n"],
                "top_acc": {str(k): v for k, v in accs.items()},
                "rescue_rate": rescue}}) + "\n")
            for tr in res["traces"]:
                tf.write(json.dumps(tr) + "\n")
        print(f"[eval] wrote per-position trace -> {args.trace_out} "
              f"({len(res['traces'])} sequences)", flush=True)

    # Free-running (native) chain acceptance (PR #34 advisor ask): the serving-side
    # accept/step. tf is the teacher-forced UPPER BOUND; native is what converts to
    # TPS. native_step1_top1 must match tf top-1 (a wiring self-check).
    native_metrics = {}
    if args.native:
        nat = evaluate_native(head, records, shift, device, dtype, rope_theta,
                              chain_k=args.native_k, max_starts=args.native_starts)
        gap = nat["native_step1_top1"] - acc
        print(f"[eval] native accept/step (K={nat['chain_k']}) = "
              f"{nat['native_accept_per_step']:.4f}  "
              f"(tokens/target-forward = {nat['native_accept_per_step'] + 1:.4f}; "
              f"starts={nat['n_starts']})", flush=True)
        print(f"[eval] native step-1 top1 = {nat['native_step1_top1']:.4f}  "
              f"(tf top1 = {acc:.4f}; self-check |Δ|={abs(gap):.4f} should be ~0)",
              flush=True)
        for s in sorted(nat["per_source"]):
            print(f"[eval]   native source {s:9s}: accept/step="
                  f"{nat['per_source'][s]:.4f} (starts={nat['per_source_n'][s]})",
                  flush=True)
        print("=" * 60, flush=True)
        native_metrics = {
            "eval/native_accept_per_step": nat["native_accept_per_step"],
            "eval/native_tokens_per_forward": nat["native_accept_per_step"] + 1,
            "eval/native_step1_top1": nat["native_step1_top1"],
            "eval/native_step1_vs_tf_gap": gap,
            "eval/native_chain_k": nat["chain_k"], "eval/native_n_starts": nat["n_starts"]}
        for s, v in nat["per_source"].items():
            native_metrics[f"eval/native_src_{s}_accept_per_step"] = v
            native_metrics[f"eval/native_src_{s}_n"] = nat["per_source_n"][s]

    out = {"eval/tf_acceptance_rate": acc, "eval/top1_acc": acc,
           "eval/rescue_rate": rescue, "eval/loss": res["loss"],
           "eval/n": res["n"], "eval/feature_shift": shift, "eval/band": band(acc),
           "eval/peak_gpu_gb": peak_gb}
    for k in range(2, args.top_k + 1):
        out[f"eval/top{k}_acc"] = accs[k]
    out.update(src_metrics)
    out.update(native_metrics)
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
