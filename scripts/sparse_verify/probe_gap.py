"""Measure the *exact* kept-winner vs best-pruned-token logit gap on real hidden.

The naive Cauchy-Schwarz certificate (``z_max > ||h||*R``) fires 0% on this model
because the embedding norms are nearly flat (R ~ 1.27) while the winning effective
projection ``z_max/||h||`` tops out at ~0.59. The naive bound on the pruned max is
therefore ~5x looser than the truth.

The question that decides whether *any* sound certificate can win: what is the
*actual* margin between the kept-set winner and the largest pruned-token logit? We
compute both exactly here (full GEMM, for measurement only) for an oracle frequency
kept set, and report ``gap/||h|| = (z_max_kept - z_max_pruned)/||h||``. If this is
comfortably positive, a tighter (e.g. low-rank residual) certificate could capture
it; if it is ~0, the lever cannot win on this model regardless of certificate.

We bracket the truth with two kept sets:
  * oracle-argmax : S = every token that is the true argmax somewhere (upper bound
    on any frequency kept set's quality; guarantees argmax in kept).
  * norm-topk-12k : the model-derived set (lower bound; argmax rarely in kept).
"""

from __future__ import annotations

import argparse

import torch

from harness_common import encode_prompt, read_sharegpt_prompts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-E4B-it")
    ap.add_argument("--num-prompts", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=256)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    print("loading model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    head = model.get_output_embeddings()
    W = head.weight.detach()
    V, H = W.shape

    captured: list[torch.Tensor] = []

    def pre_hook(module, inputs):
        x = inputs[0]
        captured.append(x.detach().reshape(-1, x.shape[-1]).to(torch.float32).cpu())

    handle = head.register_forward_pre_hook(pre_hook)
    prompts = read_sharegpt_prompts(num_prompts=args.num_prompts)
    with torch.no_grad():
        for rec in prompts:
            ids = encode_prompt(tok, rec["prompt_text"])[: args.max_tokens]
            model(input_ids=torch.tensor([ids], device="cuda"), use_cache=False)
    handle.remove()

    hidden = torch.cat(captured, dim=0).to("cuda")  # [N,H] fp32
    N = hidden.shape[0]
    hnorm = hidden.norm(dim=1)  # [N]
    Wc = W.to(torch.float32)
    norms = Wc.norm(dim=1)
    print(f"captured {N} real hidden states", flush=True)

    # Pass 1: global argmax + global max.
    full_max = torch.full((N,), -1e30, device="cuda")
    full_arg = torch.zeros((N,), dtype=torch.long, device="cuda")
    chunk = 32768
    with torch.no_grad():
        for s in range(0, V, chunk):
            e = min(s + chunk, V)
            z = hidden @ Wc[s:e].t()
            cmax, carg = z.max(dim=1)
            upd = cmax > full_max
            full_max = torch.where(upd, cmax, full_max)
            full_arg = torch.where(upd, carg + s, full_arg)

    def gap_for(name: str, kept_mask: torch.Tensor) -> None:
        kept_mask = kept_mask.to("cuda")
        # kept-winner logit and pruned-max logit, exact.
        kept_max = torch.full((N,), -1e30, device="cuda")
        pruned_max = torch.full((N,), -1e30, device="cuda")
        with torch.no_grad():
            for s in range(0, V, chunk):
                e = min(s + chunk, V)
                z = hidden @ Wc[s:e].t()  # [N, c]
                km = kept_mask[s:e]  # [c]
                zk = z.masked_fill(~km.unsqueeze(0), -1e30).max(dim=1).values
                zp = z.masked_fill(km.unsqueeze(0), -1e30).max(dim=1).values
                kept_max = torch.maximum(kept_max, zk)
                pruned_max = torch.maximum(pruned_max, zp)
        in_kept = kept_mask[full_arg]
        R = float(norms[~kept_mask.cpu()].max())
        gap = (kept_max - pruned_max) / hnorm  # >0 iff kept winner beats best pruned (per ||h||)
        gap = gap.cpu()
        # certifiable-in-principle: kept winner is the true global winner AND beats pruned max
        certifiable = in_kept.cpu() & (gap > 0)
        print(f"\n--- {name}  K={int(kept_mask.sum())}  R={R:.4f} ---", flush=True)
        print(f"  argmax_in_kept = {float(in_kept.float().mean())*100:.1f}%", flush=True)
        print(f"  certifiable-in-principle (gap>0 & in-kept) = {float(certifiable.float().mean())*100:.1f}%", flush=True)
        print("  gap/||h|| quantiles:", flush=True)
        for q in (0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9):
            print(f"    q{q:>4}: {torch.quantile(gap, q):+.4f}", flush=True)
        # how tight must a pruned-max bound be? compare actual pruned_max/||h|| vs the naive R
        ph = (pruned_max / hnorm).cpu()
        print(f"  actual pruned_max/||h||: med={ph.median():.4f} q99={torch.quantile(ph,0.99):.4f}  (naive bound R={R:.3f})", flush=True)

    # oracle frequency kept set: every true argmax token (+ padded to ~12k by norm just to set a realistic R)
    uniq = torch.unique(full_arg).cpu()
    pad = torch.topk(norms.cpu(), 12000).indices
    oracle = torch.zeros(V, dtype=torch.bool)
    oracle[uniq] = True
    oracle[pad] = True
    gap_for("oracle-argmax(+normpad)", oracle)

    normmask = torch.zeros(V, dtype=torch.bool)
    normmask[torch.topk(norms.cpu(), 12000).indices] = True
    gap_for("norm-topk-12k", normmask)


if __name__ == "__main__":
    main()
