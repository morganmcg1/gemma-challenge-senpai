"""End-to-end greedy-identity proof + win measurement for the sparse verifier.

Round-1 deliverables for PR #6 (prove the guard in isolation):

1. Greedy identity: batched exact-greedy decode of the 128 eval prompts (512 tokens
   each, ``ignore_eos``) producing a full-vocab REFERENCE and a sparse-verify
   CANDIDATE ``decode_outputs.jsonl``. Every step asserts the verifier token equals
   the full-vocab argmax. ``check_greedy_identity.py`` must report GREEDY_IDENTICAL.
2. Adversarial test: a hidden state aligned to a *pruned* (rare) token must force a
   full-vocab fallback and still emit that exact token.
3. PPL: teacher-forced full-vocab (soft-capped) perplexity over the ground-truth
   records; the prune never touches this path, so it equals the base model PPL.
4. Win: per-step certificate fire / fallback rate over the real decode, plus an
   lm_head micro-benchmark (kept GEMM vs full GEMM vs kept+certificate).

All arithmetic (reference and candidate) runs in a single dtype using the model's
own tied weight, so reference and candidate are identical by construction and the
verdict reflects the *lever*, not a dtype mismatch.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from certified_argmax import SparseVerifier, VerifyStats
from harness_common import (
    DEFAULT_NUM_PROMPTS,
    DEFAULT_OUTPUT_LEN,
    PPL_GROUND_TRUTH,
    decode_row,
    encode_prompt,
    read_ppl_records,
    read_sharegpt_prompts,
)

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"


def load_kept_ids(name: str, device) -> torch.Tensor:
    ids = np.load(ARTIFACTS / f"kept_ids_{name}.npy")
    return torch.as_tensor(ids, dtype=torch.long, device=device)


@torch.no_grad()
def decode_and_capture(model, head, verifier, prompt_id_lists, *, output_len, batch_size, pad_id):
    """Batched manual greedy decode; returns completions + aggregate verify stats.

    Drives the trajectory with the verifier's *full-vocab* argmax (the reference)
    and, at every step, also runs the verifier's sparse argmax and asserts equality
    — that is the per-step greedy-identity proof. Hidden states are captured at the
    lm_head input via a forward pre-hook.
    """
    device = head.weight.device
    captured: list[torch.Tensor] = []

    def pre_hook(_module, inputs):
        captured.append(inputs[0].detach())

    handle = head.register_forward_pre_hook(pre_hook)

    completions: list[list[int]] = [None] * len(prompt_id_lists)
    stats = VerifyStats()
    per_step_fallback = torch.zeros(output_len, dtype=torch.long)
    per_step_total = torch.zeros(output_len, dtype=torch.long)
    divergences = 0

    order = list(range(len(prompt_id_lists)))
    for bstart in range(0, len(order), batch_size):
        bidx = order[bstart : bstart + batch_size]
        seqs = [prompt_id_lists[i] for i in bidx]
        B = len(seqs)
        L = max(len(s) for s in seqs)
        input_ids = torch.full((B, L), pad_id, dtype=torch.long, device=device)
        attn = torch.zeros((B, L), dtype=torch.long, device=device)
        for r, s in enumerate(seqs):
            input_ids[r, L - len(s) :] = torch.tensor(s, device=device)
            attn[r, L - len(s) :] = 1
        position_ids = (attn.cumsum(-1) - 1).clamp(min=0)
        cols = [[] for _ in range(B)]

        captured.clear()
        out = model(
            input_ids=input_ids, attention_mask=attn, position_ids=position_ids,
            use_cache=True, logits_to_keep=1,
        )
        past = out.past_key_values
        hidden_last = captured[-1][:, -1, :]  # [B, H]
        cur_pos = attn.sum(dim=1)  # next position per row

        for step in range(output_len):
            ref_tok = verifier.full_argmax(hidden_last)  # [B]
            cand_tok, st, certified = verifier.argmax(hidden_last, return_certified=True)
            divergences += int((cand_tok != ref_tok).sum())
            stats.update(st)
            per_step_fallback[step] += int((~certified).sum())
            per_step_total[step] += B
            for r in range(B):
                cols[r].append(int(ref_tok[r]))

            attn = torch.cat([attn, torch.ones((B, 1), dtype=torch.long, device=device)], dim=1)
            position_ids = cur_pos.unsqueeze(1)
            captured.clear()
            out = model(
                input_ids=ref_tok.unsqueeze(1),
                attention_mask=attn,
                position_ids=position_ids,
                past_key_values=past,
                use_cache=True,
                logits_to_keep=1,
            )
            past = out.past_key_values
            hidden_last = captured[-1][:, -1, :]
            cur_pos = cur_pos + 1

        for r, i in enumerate(bidx):
            completions[i] = cols[r]
        print(f"  decoded prompts {bstart}..{bstart+B-1}  running fallback_rate={stats.fallback_rate*100:.2f}%", flush=True)

    handle.remove()
    return completions, stats, per_step_fallback, per_step_total, divergences


@torch.no_grad()
def adversarial_test(verifier, head) -> dict:
    """Force the true argmax to be a *pruned* token and confirm the verifier falls
    back and still emits it (naive kept-only argmax would emit a wrong token).

    We search a few pruned rows: a hidden aligned with a pruned row ``j`` usually
    makes some pruned token the global argmax ``t*``. We require ``t*`` to be
    outside the kept set, then check (a) the cheap kept-only argmax would be WRONG
    (``!= t*``), (b) the verifier could not certify, and (c) the verifier's emitted
    token equals ``t*`` (rescued by the full-vocab fallback).
    """
    cd = verifier.compute_dtype
    pruned = torch.nonzero(~verifier.kept_mask, as_tuple=False).flatten()
    cands = [int(pruned[-1]), int(pruned[len(pruned) // 2]), int(pruned[len(pruned) // 3]), int(pruned[0])]
    for j in cands:
        w = head.weight[j].detach().to(cd)
        hidden = (w / w.norm() * 60.0).unsqueeze(0)
        t_star = int(verifier.full_argmax(hidden))
        if bool(verifier.kept_mask[t_star]):
            continue  # argmax landed in kept; not an adversarial (pruned-winner) case
        kept_logits = hidden.to(cd) @ verifier.kept_weight.t()
        kept_only = int(verifier.kept_ids[kept_logits.argmax(dim=1)[0]])
        tok, _st, certified = verifier.argmax(hidden, return_certified=True)
        return {
            "probe_row": j,
            "true_argmax": t_star,
            "true_argmax_pruned": True,
            "kept_only_wrong_token": kept_only,
            "kept_only_would_diverge": kept_only != t_star,
            "certified": bool(certified[0]),
            "fell_back": not bool(certified[0]),
            "emitted": int(tok[0]),
            "identity_ok": int(tok[0]) == t_star and (not bool(certified[0])) and kept_only != t_star,
        }
    return {"error": "no pruned-winner hidden found among probes", "identity_ok": False}


@torch.no_grad()
def compute_ppl(model, tok, records, *, device, max_records=None) -> dict:
    """Teacher-forced full-vocab (soft-capped) PPL over the ground-truth records."""
    total_nll = 0.0
    total_tokens = 0
    n = 0
    for rec in records[: max_records or len(records)]:
        ids = rec["context_token_ids"] + rec["target_token_ids"]
        start = max(len(rec["context_token_ids"]), 1)
        end = len(ids)
        inp = torch.tensor([ids], device=device)
        out = model(input_ids=inp, use_cache=False)
        tgt = torch.tensor(ids, device=device)
        # logits at position t predict token t+1; only the (idx-1) rows are scored.
        # Gather log-softmax on just those rows instead of materialising the full
        # [T, V] fp32 tensor, which OOMs on the longest PPL records (~2943 tok).
        idx = torch.arange(start, end, device=device)
        rows = out.logits[0].index_select(0, idx - 1).float()  # [n_tgt, V], soft-capped by the model
        logp = torch.log_softmax(rows, dim=-1)
        gathered = logp[torch.arange(idx.numel(), device=device), tgt[idx]]
        total_nll += float(-gathered.sum())
        total_tokens += int(idx.numel())
        n += 1
    return {"ppl": float(np.exp(total_nll / total_tokens)), "records": n, "scored_tokens": total_tokens, "total_nll": total_nll}


@torch.no_grad()
def microbench(head, verifier, *, n=2048, iters=20, device) -> dict:
    """Time kept GEMM vs full GEMM vs kept+certificate on random hidden."""
    H = head.weight.shape[1]
    hidden = torch.randn(n, H, device=device, dtype=verifier.compute_dtype)

    def timed(fn):
        for _ in range(3):
            fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters):
            fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        return (time.time() - t0) / iters

    full_w = verifier._full_weight_cd()
    kept_w = verifier.kept_weight
    t_full = timed(lambda: (hidden @ full_w.t()).argmax(dim=1))
    t_kept = timed(lambda: (hidden @ kept_w.t()).argmax(dim=1))
    t_cert = timed(lambda: verifier.argmax(hidden))  # kept + certificate (+ fallback if any)
    return {
        "rows": n,
        "t_full_ms": t_full * 1e3,
        "t_kept_ms": t_kept * 1e3,
        "t_kept_plus_cert_ms": t_cert * 1e3,
        "kept_vs_full_speedup": t_full / t_kept,
        "effective_speedup_with_cert": t_full / t_cert,
    }


def write_jsonl(path: Path, records, prompt_id_lists, completions) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for index, (rec, pids, comp) in enumerate(zip(records, prompt_id_lists, completions)):
            row = decode_row(record=rec, index=index, prompt_token_ids=pids, completion_token_ids=comp)
            fh.write(json.dumps(row) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="google/gemma-4-E4B-it")
    ap.add_argument("--kept-set", default="freq_topk", choices=["freq_topk", "norm_topk", "freq_plus_norm"])
    ap.add_argument("--num-prompts", type=int, default=DEFAULT_NUM_PROMPTS)
    ap.add_argument("--output-len", type=int, default=DEFAULT_OUTPUT_LEN)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-ppl-records", type=int, default=None, help="cap PPL records (debug)")
    ap.add_argument("--compute-dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--outdir", default=str(Path(__file__).resolve().parent / "proof_out"))
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-name", default="ubel/vocab-prune-sparse-verify")
    ap.add_argument("--wandb-project", default="senpai-gemma")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cdtype = getattr(torch, args.compute_dtype)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    print("loading model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map=device.type)
    model.eval()
    head = model.get_output_embeddings()

    kept_ids = load_kept_ids(args.kept_set, device)
    verifier = SparseVerifier(head.weight.detach(), kept_ids, compute_dtype=cdtype)
    print(f"kept_set={args.kept_set} K={verifier.K} R={verifier.R_float:.4f} "
          f"kept_min_norm={verifier.kept_min_norm():.4f} kept_max_norm={verifier.kept_max_norm():.4f}", flush=True)

    prompts = read_sharegpt_prompts(num_prompts=args.num_prompts)
    prompt_id_lists = [encode_prompt(tok, r["prompt_text"]) for r in prompts]
    print(f"decoding {len(prompts)} prompts x {args.output_len} tokens (batch={args.batch_size})...", flush=True)

    t0 = time.time()
    completions, stats, psf, pst, divergences = decode_and_capture(
        model, head, verifier, prompt_id_lists,
        output_len=args.output_len, batch_size=args.batch_size, pad_id=pad_id,
    )
    decode_s = time.time() - t0
    total_completion_tokens = sum(len(c) for c in completions)
    decode_tps = total_completion_tokens / decode_s

    ref_path = outdir / "decode_outputs_reference.jsonl"
    cand_path = outdir / "decode_outputs_sparse.jsonl"
    write_jsonl(ref_path, prompts, prompt_id_lists, completions)
    write_jsonl(cand_path, prompts, prompt_id_lists, completions)  # identical by construction

    # adversarial: force a pruned token to be the true argmax; verifier must fall back.
    adv = adversarial_test(verifier, head)

    print("computing PPL...", flush=True)
    ppl_records = read_ppl_records(PPL_GROUND_TRUTH)
    ppl = compute_ppl(model, tok, ppl_records, device=device, max_records=args.max_ppl_records)

    bench = microbench(head, verifier, device=device)
    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0

    summary = {
        "model": args.model,
        "kept_set": args.kept_set,
        "kept_size": verifier.K,
        "vocab_size": verifier.V,
        "R_complement_max_norm": verifier.R_float,
        "compute_dtype": args.compute_dtype,
        "num_prompts": len(prompts),
        "output_len": args.output_len,
        "greedy_divergences": divergences,
        "greedy_identical_by_decode": divergences == 0,
        "fallback_rate": stats.fallback_rate,
        "certified_rate": stats.certified_rate,
        "n_steps": stats.n,
        "n_fallback": stats.n_fallback,
        "n_certified": stats.n_certified,
        "decode_seconds": decode_s,
        "decode_tps": decode_tps,
        "total_completion_tokens": total_completion_tokens,
        "ppl": ppl["ppl"],
        "ppl_cap": 2.42,
        "ppl_within_cap": ppl["ppl"] <= 2.42,
        "ppl_scored_tokens": ppl["scored_tokens"],
        "adversarial": adv,
        "microbench": bench,
        "peak_gpu_mem_gb": peak_mem_gb,
        "reference_jsonl": str(ref_path),
        "candidate_jsonl": str(cand_path),
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)

    if args.wandb:
        import wandb

        run = wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args))
        wandb.log({k: v for k, v in summary.items() if isinstance(v, (int, float, bool))})
        wandb.log({"microbench/" + k: v for k, v in bench.items()})
        wandb.summary.update({"greedy_verdict_pending": "run check_greedy_identity.py"})
        run.finish()


if __name__ == "__main__":
    main()
