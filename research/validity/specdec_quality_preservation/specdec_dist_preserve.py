#!/usr/bin/env python
"""PR #505 leg-2: empirical sampled-distribution preservation of the DEPLOYED
surgical-357 spec-dec acceptance rule under temperature sampling.

We drive the EXACT pinned vLLM rejection sampler that the surgical-357 serve
uses (server-venv 0.22.1rc1.dev307+g3e8afdf78, dixie-patched -- the patch only
adds an all_greedy fast-path short-circuit; rejection_sample()/random kernel/
recovered kernel are stock). Run it in the deployed MTP config:
  draft_sample_method='greedy' (default) -> greedy deterministic draft,
  draft_probs=None (NO_DRAFT_PROBS),
  rejection_sample_method='standard'.

For a deterministic (greedy) draft x_d the standard rejection rule is:
  accept x_d w.p. min(1, p(x_d)),  else resample from p restricted to {y != x_d}.
This is EXACTLY distribution-preserving: output ~ p. We confirm empirically by
histogramming the first emitted token over many trials and comparing to p via
total-variation distance and KL(p || p_hat), against the Monte-Carlo noise floor
(TV of an i.i.d. multinomial draw of the same size).

Usage:
  specdec_dist_preserve.py            # synthetic shapes (Test 1)
  specdec_dist_preserve.py --logits real_logits.pt --out real_results.json
"""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import torch

from vllm.v1.sample.rejection_sampler import rejection_sample, PLACEHOLDER_TOKEN_ID

DEVICE = torch.device("cuda")


def deployed_first_token_hist(
    p: torch.Tensor, draft_token_id: int, M: int, seed: int
) -> torch.Tensor:
    """Run the deployed rejection_sample M times on target dist p with a fixed
    greedy/deterministic draft token; return the empirical first-token counts.

    target_logits passed to rejection_sample are the ALREADY temp/top-k/top-p
    processed logits (that is what apply_sampling_constraints produces upstream),
    so softmax(target_logits) == p. We set target_logits = log(p)."""
    vocab = p.numel()
    torch.manual_seed(seed)
    # softmax(log p) == p exactly (p sums to 1); zeros -> -inf -> prob 0.
    logp = torch.log(p.clamp_min(0)).to(torch.float32)
    target_logits = logp.unsqueeze(0).expand(M, vocab).contiguous().to(DEVICE)
    draft_token_ids = torch.full((M,), int(draft_token_id), dtype=torch.int32, device=DEVICE)
    num_draft_tokens = [1] * M
    cu_num_draft_tokens = torch.arange(1, M + 1, dtype=torch.int32, device=DEVICE)
    bonus = int(torch.argmax(p).item())
    bonus_token_ids = torch.full((M, 1), bonus, dtype=torch.int32, device=DEVICE)
    sm = SimpleNamespace(
        all_greedy=False,
        all_random=True,
        temperature=torch.ones(M, dtype=torch.float32, device=DEVICE),  # !=0 marker
        generators={},
    )
    out = rejection_sample(
        draft_token_ids,
        num_draft_tokens,
        1,  # max_spec_len
        cu_num_draft_tokens,
        None,  # draft_probs -> NO_DRAFT_PROBS (deployed greedy-draft MTP)
        target_logits,
        bonus_token_ids,
        sm,
    )
    first = out[:, 0].to(torch.int64)
    assert int((first == PLACEHOLDER_TOKEN_ID).sum()) == 0, "placeholder in first token"
    counts = torch.bincount(first.cpu(), minlength=vocab).to(torch.float64)
    return counts


def tv(a: torch.Tensor, b: torch.Tensor) -> float:
    return 0.5 * float(torch.abs(a - b).sum())


def kl(p: torch.Tensor, q: torch.Tensor) -> float:
    # KL(p || q), support of p; q smoothed to avoid div-by-zero.
    eps = 1e-12
    mask = p > 0
    pp = p[mask]
    qq = q[mask].clamp_min(eps)
    return float((pp * (pp / qq).log()).sum())


def eval_case(p: torch.Tensor, M: int, seed: int, label: str) -> dict:
    p = (p / p.sum()).to(torch.float64)
    vocab = p.numel()
    greedy_draft = int(torch.argmax(p).item())
    # adversarial draft: a low-but-nonzero-prob token (worst case for over-accept)
    support = torch.nonzero(p > 0).flatten()
    adv_draft = int(support[torch.argmin(p[support])].item())

    res = {"label": label, "vocab": vocab, "M": M, "p_max": float(p.max()),
           "p_entropy_nats": float(-(p[p > 0] * p[p > 0].log()).sum()),
           "greedy_draft": greedy_draft, "adv_draft": adv_draft}

    for name, dtok in (("greedy_draft", greedy_draft), ("adv_draft", adv_draft)):
        counts = deployed_first_token_hist(p.to(torch.float32), dtok, M, seed)
        phat = counts / counts.sum()
        # Monte-Carlo noise floor: i.i.d. multinomial draw of same size from p.
        g = torch.Generator().manual_seed(seed + 7)
        iid = torch.multinomial(p, M, replacement=True, generator=g)
        iid_counts = torch.bincount(iid, minlength=vocab).to(torch.float64)
        piid = iid_counts / iid_counts.sum()
        res[name] = {
            "tv_deployed_vs_p": tv(phat, p),
            "tv_iid_noise_floor": tv(piid, p),
            "kl_p_given_deployed": kl(p, phat),
            "kl_p_given_iid": kl(p, piid),
            "accept_rate_top": float((counts[greedy_draft] / counts.sum())),
        }
    return res


def synthetic_suite(M: int, seed: int) -> list[dict]:
    cases = []
    # near-deterministic confident MCQ answer-letter distribution
    cases.append(("confident_mcq", torch.tensor([0.97, 0.02, 0.006, 0.004])))
    # moderate two-way
    cases.append(("two_way_60_40", torch.tensor([0.6, 0.4])))
    # graded 4-way (typical letter spread)
    cases.append(("graded_4way", torch.tensor([0.45, 0.30, 0.15, 0.10])))
    # peaked power-law over vocab=64 (long numeric-token tail, hard_ood-like)
    v = 64
    pl = 1.0 / torch.arange(1, v + 1, dtype=torch.float64) ** 1.5
    cases.append(("powerlaw_v64", pl))
    # broad-ish over vocab=256 (higher-entropy regime stress)
    v2 = 256
    broad = torch.softmax(0.5 * torch.randn(v2, generator=torch.Generator().manual_seed(0)), dim=0).double()
    cases.append(("broad_v256", broad))

    out = []
    for label, p in cases:
        out.append(eval_case(p, M, seed, label))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--logits", type=str, default=None,
                    help="optional .pt with dict{'p': [N,vocab] target sampling dists, 'meta': ...}")
    ap.add_argument("--out", type=str, default="synthetic_results.json")
    args = ap.parse_args()

    if args.logits is None:
        results = synthetic_suite(args.M, args.seed)
        kind = "synthetic"
    else:
        blob = torch.load(args.logits)
        P = blob["p"]  # [N, vocab] each row a target sampling distribution
        meta = blob.get("meta", [{}] * len(P))
        results = []
        for i in range(len(P)):
            r = eval_case(P[i].double(), args.M, args.seed + i, meta[i].get("id", f"row{i}"))
            r["meta"] = meta[i]
            results.append(r)
        kind = "real_logits"

    # aggregate
    tv_dep = [c[d]["tv_deployed_vs_p"] for c in results for d in ("greedy_draft", "adv_draft")]
    tv_floor = [c[d]["tv_iid_noise_floor"] for c in results for d in ("greedy_draft", "adv_draft")]
    kl_dep = [c[d]["kl_p_given_deployed"] for c in results for d in ("greedy_draft", "adv_draft")]
    summary = {
        "kind": kind, "M": args.M, "n_cases": len(results),
        "mean_tv_deployed_vs_p": sum(tv_dep) / len(tv_dep),
        "max_tv_deployed_vs_p": max(tv_dep),
        "mean_tv_iid_noise_floor": sum(tv_floor) / len(tv_floor),
        "mean_kl_p_given_deployed": sum(kl_dep) / len(kl_dep),
        "max_kl_p_given_deployed": max(kl_dep),
    }
    blob_out = {"summary": summary, "cases": results}
    with open(args.out, "w") as f:
        json.dump(blob_out, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"[wrote] {args.out}")


if __name__ == "__main__":
    main()
