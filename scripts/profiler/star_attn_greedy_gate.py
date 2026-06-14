#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #93 - Star-attention greedy-equivalence gate.

QUESTION
--------
land #71 / chiku-inu implement the tree-verify mask via a triton STAR-ATTENTION
path validated to relerr ~1e-3 vs a dense-mask reference. relerr 1e-3 is NOT
bit-exact. The greedy-identity contract (program.md 27-28) requires the served
greedy token sequence to be bit-for-bit identical to plain greedy AR decode of
the submitted checkpoint. Gate: does a ~1e-3 relative perturbation of the
ATTENTION OUTPUT flip the greedy argmax at any decode position?

METHOD (LOCAL, CPU/1-GPU, no HF launch; measurement only - served files never
touched)
-------
Reference frame = the contract's own reference: plain greedy AR decode of the
deployed int4 verifier ``/tmp/osoi5-v0-baked`` (the exact baked checkpoint #86
served; lm_head is the deployed pck04 16k keepset; the served argmax is over
bf16 logits via FUSED_SPARSE_ARGMAX). We teacher-force the #86 128x512 greedy
corpus (one forward per prompt over [prompt|completion]) and read the verifier's
final logits at every completion position.

  Step 1 (margin): per position, fp32 top1-top2 final-logit margin (the lm_head
    decompresses to bf16, so fp32 logits = fp32(hidden) @ fp32(W) + softcap). We
    ALSO report the bf16-readout margin + bf16 tie fraction, because the deployed
    decision argmaxes over bf16 logits (sub-bf16-ULP margins are deployment ties).

  Step 2 (perturbation): inject a controlled per-row relative Gaussian
    perturbation of magnitude eps into the o_proj INPUT (the attention context
    vector = exactly what the star-attention kernel produces) at EVERY layer,
    computed in fp32 and cast back to the deployed bf16 (so realized relerr
    reflects what a bf16 attention kernel can actually inject). Sweep
    eps in {1e-4,1e-3,1e-2} x several seeds; greedy_flip_rate = fraction of the
    128x512 bf16 argmax tokens that differ from the eps=0 baseline.

PRIMARY metric: greedy_flip_rate_at_1e3. TEST metric: min_greedy_margin_p1 (1st-
percentile fp32 top1-top2 margin in units of top-1 logit magnitude).
GREEN if flip_rate_at_1e3==0; RED otherwise (land #71 needs fp32 accumulation).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
MODEL = os.environ.get("GATE_MODEL", "/tmp/osoi5-v0-baked")
CORPUS = ROOT / "research/rank_coverage/pr86/decode_rank_coverage.jsonl"
OUT_DIR = ROOT / "research/star_attn_gate"

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# perturbation control read by the o_proj forward_pre_hooks
_PERT = {"on": False, "eps": 0.0, "gen": None, "relerr_sum": 0.0, "relerr_n": 0}


def _make_hook():
    def hook(mod, args):
        if not _PERT["on"]:
            return None
        a = args[0]
        af = a.float()
        rn = af.norm(dim=-1, keepdim=True)  # per-row (per-token) norm
        z = torch.randn(af.shape, generator=_PERT["gen"], device=af.device, dtype=af.dtype)
        delta = _PERT["eps"] * rn / math.sqrt(af.shape[-1]) * z
        a_pert = (af + delta).to(a.dtype)
        # accumulate realized relerr (post cast-back) for reporting
        with torch.no_grad():
            _PERT["relerr_sum"] += ((a_pert.float() - af).norm() / af.norm().clamp_min(1e-9)).item()
            _PERT["relerr_n"] += 1
        if len(args) == 1:
            return (a_pert,)
        return (a_pert, *args[1:])
    return hook


def load():
    from transformers import Gemma4ForConditionalGeneration
    t0 = time.time()
    model = Gemma4ForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map={"": "cuda:0"})
    model.eval()
    layers = model.model.language_model.layers
    hooks = [l.self_attn.o_proj.register_forward_pre_hook(_make_hook()) for l in layers]
    softcap = float(model.config.text_config.final_logit_softcapping)
    keep = json.load(open(os.path.join(MODEL, "pck04_keepset.json")))
    keep_t = torch.tensor(keep["keep_ids"], device="cuda")
    print(f"[gate] loaded {time.time()-t0:.1f}s n_layers={len(layers)} "
          f"attn={model.config.text_config._attn_implementation} softcap={softcap} "
          f"K={keep['pruned_vocab_K']} hooks={len(hooks)}")
    return model, keep_t, softcap


@torch.no_grad()
def run_forward(model, ids, start, L, softcap, eps=0.0, seed=0):
    """One forward (clean if eps==0 else perturbed). Returns dict with both the
    deployed bf16 argmax and the true-fp32 argmax, plus margins, per position.

    The deployed greedy decision (serve.py:410) is argmax over BF16 logits, so
    bf16_arg is the contract-faithful token. fp32_arg recomputes logits in fp32
    (lm_head decompresses to bf16 -> fp32 matmul) to decompose bf16-tie flips
    from genuine propagation flips.
    """
    cap = {}
    def lmhook(mod, args):
        cap["h"] = args[0].detach()
        return None
    hh = model.lm_head.register_forward_pre_hook(lmhook)
    if eps > 0:
        gen = torch.Generator(device="cuda")
        gen.manual_seed(seed)
        _PERT.update(on=True, eps=eps, gen=gen, relerr_sum=0.0, relerr_n=0)
    else:
        _PERT["on"] = False
    out = model(input_ids=ids, use_cache=False)
    _PERT["on"] = False
    hh.remove()
    bf = out.logits[:, start:start + L].float()            # (B,L,K) softcapped, bf16-valued
    bf_top2 = bf.topk(2, dim=-1)
    bf_arg = bf_top2.indices[..., 0]
    bf_margin = bf_top2.values[..., 0] - bf_top2.values[..., 1]
    h = cap["h"][:, start:start + L].float()               # (B,L,hidden)
    W = model.lm_head.weight.float()
    b = model.lm_head.bias.float() if getattr(model.lm_head, "bias", None) is not None else None
    fp = h @ W.t()
    if b is not None:
        fp = fp + b
    fp = softcap * torch.tanh(fp / softcap)
    fp_top2 = fp.topk(2, dim=-1)
    fp_arg = fp_top2.indices[..., 0]
    fp_margin = fp_top2.values[..., 0] - fp_top2.values[..., 1]
    fp_top1mag = fp_top2.values[..., 0].abs()
    relerr = _PERT["relerr_sum"] / max(_PERT["relerr_n"], 1) if eps > 0 else 0.0
    return {"bf_arg": bf_arg, "fp_arg": fp_arg, "fp_margin": fp_margin,
            "fp_top1mag": fp_top1mag, "bf_margin": bf_margin, "relerr": relerr}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-prompts", type=int, default=128)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--epsilons", type=str, default="1e-4,1e-3,1e-2")
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4")
    ap.add_argument("--out", type=str, default=str(OUT_DIR / "gate_results.json"))
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-name", type=str, default="wirbel/star-attn-greedy-gate")
    ap.add_argument("--wandb-group", type=str, default="star-attn-greedy-gate")
    args = ap.parse_args()
    epsilons = [float(x) for x in args.epsilons.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]

    recs = [json.loads(l) for l in open(CORPUS)][: args.num_prompts]
    model, keep_t, softcap = load()

    # accumulators
    fp_margins = []          # fp32 top1-top2 margin (abs)
    fp_relmargins = []       # margin / top1mag
    bf_margins = []          # bf16-readout margin
    clean_args = []          # bf16 clean argmax (head idx)
    gold_real = []           # served greedy real-vocab tokens
    clean_real = []          # clean argmax mapped to real vocab
    # per (eps, seed): boolean flip vector accumulated as count
    flip_counts = {e: {s: 0 for s in seeds} for e in epsilons}        # bf16 (deployed)
    flip_counts_fp32 = {e: {s: 0 for s in seeds} for e in epsilons}   # fp32 (true ranking)
    relerr_meas = {e: [] for e in epsilons}
    # noise-floor control: a SECOND unperturbed pass per batch. The forward is
    # deterministic (verified bit-identical), so this must be 0 - it proves every
    # measured flip is perturbation-attributable, not run-to-run nondeterminism.
    noise_floor_bf = 0
    noise_floor_fp = 0
    n_positions = 0
    # store per-position fp margin to analyze WHERE flips happen (eps=1e-3, seed0)
    flip_margin_at_1e3 = []  # fp32 margins of positions that flip at 1e-3 (any seed)

    t0 = time.time()
    fwd_times = []
    for bi in range(0, len(recs), args.batch):
        batch = recs[bi:bi + args.batch]
        # all #86 records are prompt256+compl512; pad-safe: group by equal length
        plen = len(batch[0]["prompt_token_ids"])
        L = len(batch[0]["completion_token_ids"])
        assert all(len(r["prompt_token_ids"]) == plen and len(r["completion_token_ids"]) == L
                   for r in batch), "ragged batch"
        ids = torch.tensor([r["prompt_token_ids"] + r["completion_token_ids"] for r in batch],
                           device="cuda")
        start = plen - 1
        torch.cuda.synchronize(); tt = time.time()
        clean = run_forward(model, ids, start, L, softcap, eps=0.0)
        torch.cuda.synchronize(); fwd_times.append(time.time() - tt)
        bf_arg = clean["bf_arg"]        # (B,L) deployed bf16 argmax (contract-faithful)
        fp_arg = clean["fp_arg"]        # (B,L) true fp32 argmax (decomposition baseline)
        fp_margin = clean["fp_margin"]
        fp_top1mag = clean["fp_top1mag"]
        bf_margin = clean["bf_margin"]
        B = ids.shape[0]
        n_positions += B * L
        fp_margins.append(fp_margin.reshape(-1).cpu())
        fp_relmargins.append((fp_margin / fp_top1mag.clamp_min(1e-6)).reshape(-1).cpu())
        bf_margins.append(bf_margin.reshape(-1).cpu())
        clean_args.append(bf_arg.reshape(-1).cpu())
        clean_real_b = keep_t[bf_arg]  # (B,L)
        for j, r in enumerate(batch):
            gold_real.extend(r["completion_token_ids"])
            clean_real.extend(clean_real_b[j].cpu().tolist())
        # noise-floor control: second unperturbed pass (must be 0)
        clean2 = run_forward(model, ids, start, L, softcap, eps=0.0)
        noise_floor_bf += int((clean2["bf_arg"] != bf_arg).sum().item())
        noise_floor_fp += int((clean2["fp_arg"] != fp_arg).sum().item())
        # perturbation passes: bf16 argmax flip (deployment-faithful, PRIMARY) and
        # fp32 argmax flip (true-ranking, decomposes bf16-tie noise from propagation)
        flip_any_1e3 = torch.zeros(B, L, dtype=torch.bool, device="cuda")
        for e in epsilons:
            for s in seeds:
                pert = run_forward(model, ids, start, L, softcap, eps=e, seed=s)
                bf_flips = (pert["bf_arg"] != bf_arg)
                fp_flips = (pert["fp_arg"] != fp_arg)
                flip_counts[e][s] += int(bf_flips.sum().item())
                flip_counts_fp32[e][s] += int(fp_flips.sum().item())
                relerr_meas[e].append(pert["relerr"])
                if abs(e - 1e-3) < 1e-12:
                    flip_any_1e3 |= bf_flips
        if flip_any_1e3.any():
            flip_margin_at_1e3.append(fp_margin[flip_any_1e3].cpu())
        if (bi // args.batch) % 4 == 0:
            print(f"[gate] batch {bi//args.batch+1}/{math.ceil(len(recs)/args.batch)} "
                  f"pos={n_positions} fwd~{sum(fwd_times)/len(fwd_times):.2f}s "
                  f"elapsed={time.time()-t0:.0f}s", flush=True)

    fp_margins = torch.cat(fp_margins)
    fp_relmargins = torch.cat(fp_relmargins)
    bf_margins = torch.cat(bf_margins)
    clean_args = torch.cat(clean_args)
    gold_real_t = torch.tensor(gold_real)
    clean_real_t = torch.tensor(clean_real)

    def pct(t, q):
        k = max(1, int(q * len(t)))
        return float(t.kthvalue(k).values)

    # ----- Step 1 metrics -----
    step1 = {
        "fp32_margin_median": float(fp_margins.median()),
        "fp32_margin_p1": pct(fp_margins, 0.01),
        "fp32_margin_p0.1": pct(fp_margins, 0.001),
        "fp32_margin_min": float(fp_margins.min()),
        "fp32_relmargin_median": float(fp_relmargins.median()),
        "min_greedy_margin_p1": pct(fp_relmargins, 0.01),     # TEST METRIC
        "min_greedy_margin_p0.1": pct(fp_relmargins, 0.001),
        "frac_relmargin_lt_1e2": float((fp_relmargins < 1e-2).float().mean()),
        "frac_relmargin_lt_1e3": float((fp_relmargins < 1e-3).float().mean()),
        "frac_relmargin_lt_1e4": float((fp_relmargins < 1e-4).float().mean()),
        "frac_fp32_exact_tie": float((fp_margins == 0).float().mean()),
        "bf16_margin_median": float(bf_margins.median()),
        "frac_bf16_tie": float((bf_margins == 0).float().mean()),       # deployment ties
        "frac_bf16_le_1ulp": float((bf_margins <= 0.1251).float().mean()),
        "n_positions": int(n_positions),
        "eager_vs_served_mismatch_rate": float((clean_real_t != gold_real_t).float().mean()),
    }

    # ----- Step 2 metrics -----
    step2 = {"per_eps": {}}
    first_flip_eps = None
    for e in epsilons:
        rates = [flip_counts[e][s] / n_positions for s in seeds]
        mean_rate = sum(rates) / len(rates)
        rates_fp32 = [flip_counts_fp32[e][s] / n_positions for s in seeds]
        mean_rate_fp32 = sum(rates_fp32) / len(rates_fp32)
        step2["per_eps"][f"{e:.0e}"] = {
            "greedy_flip_rate_mean": mean_rate,
            "greedy_flip_rate_max": max(rates),
            "greedy_flip_count_total": sum(flip_counts[e][s] for s in seeds),
            "fp32_flip_rate_mean": mean_rate_fp32,
            "fp32_flip_rate_max": max(rates_fp32),
            "fp32_flip_count_total": sum(flip_counts_fp32[e][s] for s in seeds),
            "n_seeds": len(seeds),
            "realized_relerr_mean": sum(relerr_meas[e]) / max(len(relerr_meas[e]), 1),
            "flips_per_seed": {str(s): flip_counts[e][s] for s in seeds},
        }
        if mean_rate > 0 and first_flip_eps is None:
            first_flip_eps = e
    # primary
    e3 = next(x for x in epsilons if abs(x - 1e-3) < 1e-12)
    primary = step2["per_eps"][f"{e3:.0e}"]["greedy_flip_rate_mean"]
    step2["greedy_flip_rate_at_1e3"] = primary           # PRIMARY METRIC
    step2["fp32_flip_rate_at_1e3"] = step2["per_eps"][f"{e3:.0e}"]["fp32_flip_rate_mean"]
    step2["noise_floor_bf16_rate"] = noise_floor_bf / n_positions   # must be ~0
    step2["noise_floor_fp32_rate"] = noise_floor_fp / n_positions
    step2["eps_first_flip"] = first_flip_eps
    if flip_margin_at_1e3:
        fm = torch.cat(flip_margin_at_1e3)
        step2["flip_at_1e3_fp32_margin_median"] = float(fm.median())
        step2["flip_at_1e3_fp32_margin_max"] = float(fm.max())
        step2["flip_at_1e3_count_unique_positions"] = int(len(fm))

    verdict = "GREEN" if primary == 0 else "RED"
    result = {
        "verdict": verdict,
        "primary_metric": {"name": "greedy_flip_rate_at_1e3", "value": primary},
        "test_metric": {"name": "min_greedy_margin_p1", "value": step1["min_greedy_margin_p1"]},
        "step1_margin": step1,
        "step2_flip": step2,
        "config": {
            "model": MODEL, "num_prompts": len(recs), "epsilons": epsilons, "seeds": seeds,
            "perturb_target": "o_proj_input(attention context, per-row relerr, cast-back bf16)",
            "corpus": str(CORPUS), "softcap": softcap,
            "mean_fwd_s": sum(fwd_times) / len(fwd_times),
            "elapsed_s": time.time() - t0,
            "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9,
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print("\n===== GATE RESULT =====")
    print(json.dumps(result, indent=2))

    if args.wandb:
        import wandb
        run = wandb.init(project="gemma-challenge-senpai", entity="wandb-applied-ai-team",
                         name=args.wandb_name, group=args.wandb_group,
                         config=result["config"] | {"pr": 93, "verdict": verdict})
        flat = {f"step1/{k}": v for k, v in step1.items()}
        flat.update({"verdict_red": int(verdict == "RED"),
                     "primary/greedy_flip_rate_at_1e3": primary,
                     "primary/fp32_flip_rate_at_1e3": step2["fp32_flip_rate_at_1e3"],
                     "control/noise_floor_bf16_rate": step2["noise_floor_bf16_rate"],
                     "control/noise_floor_fp32_rate": step2["noise_floor_fp32_rate"],
                     "test/min_greedy_margin_p1": step1["min_greedy_margin_p1"]})
        for e in epsilons:
            d = step2["per_eps"][f"{e:.0e}"]
            flat[f"step2/flip_rate_mean_{e:.0e}"] = d["greedy_flip_rate_mean"]
            flat[f"step2/flip_rate_max_{e:.0e}"] = d["greedy_flip_rate_max"]
            flat[f"step2/fp32_flip_rate_mean_{e:.0e}"] = d["fp32_flip_rate_mean"]
            flat[f"step2/realized_relerr_{e:.0e}"] = d["realized_relerr_mean"]
        wandb.log(flat)
        wandb.summary.update(flat)
        run.finish()
        print(f"[gate] wandb run: {run.id}")


if __name__ == "__main__":
    main()
