"""Empirical certificate-feasibility probe on real Gemma hidden states.

The greedy-safe certificate fires for a step iff ``z_max_kept > ||h|| * R`` with
``R = max_{j not in S} ||W_j||``. Because ``||h||`` cancels, this is equivalent to
``(z_max_kept / ||h||) > R``: the winning kept token's *effective projection* must
exceed the largest complement row norm. This script measures the real distribution
of that projection on actual post-final-norm hidden states, so we can predict the
fallback rate before running the full proof.

It loads the model, hooks the lm_head input, runs prefill over a handful of real
eval prompts, and reports, for several kept-set constructions, how often the
certificate would fire and how often the true argmax lands in the kept set.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from harness_common import encode_prompt, read_sharegpt_prompts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-E4B-it")
    ap.add_argument("--num-prompts", type=int, default=12)
    ap.add_argument("--max-tokens", type=int, default=256, help="cap prefill length per prompt")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    print("loading model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    head = model.get_output_embeddings()
    print("output head:", type(head).__name__, "weight", tuple(head.weight.shape), flush=True)
    W = head.weight.detach()  # [V, H] tied embedding, on cuda, bf16
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
            inp = torch.tensor([ids], device="cuda")
            model(input_ids=inp, use_cache=False)
    handle.remove()

    hidden = torch.cat(captured, dim=0)  # [N, H] fp32 cpu
    N = hidden.shape[0]
    print(f"captured {N} real hidden states", flush=True)

    # Row norms of the full vocab (fp32).
    norms = W.to(torch.float32).norm(dim=1).cpu()  # [V]

    # Compute full logits in chunks to get the true argmax + max-logit per position.
    hid_cuda = hidden.to("cuda")
    hnorm = hid_cuda.norm(dim=1)  # [N]
    full_max = torch.full((N,), -1e30, device="cuda")
    full_arg = torch.zeros((N,), dtype=torch.long, device="cuda")
    Wc = W.to(torch.float32)
    chunk = 32768
    with torch.no_grad():
        for s in range(0, V, chunk):
            e = min(s + chunk, V)
            z = hid_cuda @ Wc[s:e].t()  # [N, chunk]
            cmax, carg = z.max(dim=1)
            upd = cmax > full_max
            full_max = torch.where(upd, cmax, full_max)
            full_arg = torch.where(upd, carg + s, full_arg)
    proj = (full_max / hnorm).cpu()  # [N] = z_max / ||h|| = winning effective projection
    full_arg = full_arg.cpu()

    print("\n=== winning effective projection (z_max/||h||) over real hidden ===", flush=True)
    for q in (0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99):
        print(f"  q{q:>4}: {torch.quantile(proj, q):.4f}")
    print(f"  min={proj.min():.4f} max={proj.max():.4f} mean={proj.mean():.4f}")

    # For each kept-set construction, measure certificate fire rate + argmax-in-kept.
    def report(name: str, kept_ids: torch.Tensor) -> None:
        mask = torch.zeros(V, dtype=torch.bool)
        mask[kept_ids] = True
        R = float(norms[~mask].max()) if (~mask).any() else 0.0
        in_kept = mask[full_arg]  # argmax in kept set?
        # certificate fires iff z_max_kept > ||h||*R. z_max_kept == full_max when argmax in kept.
        # When argmax not in kept, z_max_kept < full_max and certificate cannot certify a wrong tok,
        # so it necessarily falls back. Upper-bound the fire rate by (argmax in kept) & (proj > R).
        fires = in_kept & (proj > R)
        print(
            f"  {name:16s} K={int(mask.sum()):6d} R={R:.4f} "
            f"argmax_in_kept={float(in_kept.float().mean())*100:5.1f}% "
            f"cert_fire={float(fires.float().mean())*100:5.1f}%",
            flush=True,
        )

    print("\n=== kept-set constructions (certificate fire rate is the win) ===", flush=True)
    for k in (8000, 12000, 24000, 50000):
        report(f"norm-topk-{k}", torch.topk(norms, k).indices)
    # frequency proxy: the tokens actually argmaxed here (overfit upper bound on freq-topk)
    uniq_arg = torch.unique(full_arg)
    print(f"\n(distinct argmax tokens across {N} positions: {uniq_arg.numel()})", flush=True)


if __name__ == "__main__":
    main()
