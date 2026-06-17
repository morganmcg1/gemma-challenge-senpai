#!/usr/bin/env python
"""STRETCH (feasible variant): cumulative restore-k KL-RECOVERY curve (#586).

DIAGNOSTIC ONLY (analysis_only=true, official_tps=0).

The generation-based AIME pass@1 restore-k curve is INFEASIBLE at fidelity: the
AIME harness uses max_tokens=3072 (long reasoning) to produce the cited int4=0.1167
/ unquantized=0.400 anchors, and at ~15-20 tok/s on eager dense bf16 (single A10G,
no Marlin/flash) a 24-problem x 5-k sweep at 3072 tokens is ~tens of GPU-hours; a
truncated 512-token cap leaves BOTH endpoints near-zero (truncated reasoning -> no
boxed answer) -> a flat, uninformative curve. See report.

So we run the same-fidelity LOGIT-level joint test instead: rank the 42 body layers
by the CLEAN quantize-from-bf16 sensitivity (quantize_result.json, all-positive),
then cumulatively restore the top-k to bf16 in the int4-dense working model and
measure how fast the teacher-forced KL gap toward full-bf16 collapses:
    D_k    = KL(bf16 || int4 with top-k quantize-ranked layers restored)
    rec_k  = (D_int4 - D_k) / D_int4          # 0 at k=0, ~1 at k=42
A SHARP rise at small k => concentrated (floor liftable by a few layers);
a GRADUAL rise => diffuse (floor is an emergent co-adapted whole). Unlike the
single-layer analysis, the cumulative sweep captures inter-layer interactions
(QAT co-adaptation), so non-monotonicity (KL rising as a co-adapted layer is
restored in isolation) is itself diagnostic. Ranking by the quantize direction
gives concentration its BEST shot: if even this ordering recovers gradually,
diffuse is robust.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, CompressedTensorsConfig

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))
from profile_sensitivity import (  # noqa: E402  (reuse validated helpers)
    BF16_DIR, INT4_ID, DEVICE, get_layers, layer_shared_sd,
    teacher_forced_logprobs, build_batch,
)
from research.downstream_quality_aime.aime_eval import load_aime  # noqa: E402

CONC_K = 5          # "concentrated" if >=90% KL recovered by restoring <=5 layers (~12% of body)
REC_BAR = 0.90


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="2024,2025")
    ap.add_argument("--n-problems", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=320)
    ap.add_argument("--min-new-tokens", type=int, default=8)
    ap.add_argument("--k-grid", default="0,1,2,3,5,8,10,15,20,30,42")
    ap.add_argument("--ranking", default=str(HERE / "quantize_result.json"))
    ap.add_argument("--rank-key", default="ranked_layers_by_d_kl",
                    help="ranked_layers_by_d_kl (clean quantize dir) or ranked_layers_by_s_kl (restore dir)")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--out", default=str(HERE / "restore_k_kl_result.json"))
    args = ap.parse_args()

    t0 = time.time()
    rank = json.loads(Path(args.ranking).read_text())[args.rank_key]
    ranked_layers = [int(li) for li, _ in rank]   # descending sensitivity

    tok = AutoTokenizer.from_pretrained(INT4_ID, trust_remote_code=True)
    pall = load_aime([y.strip() for y in args.years.split(",")], limit=None)
    if args.n_problems and args.n_problems < len(pall):
        stride = len(pall) / args.n_problems
        idx = sorted({int(i * stride) for i in range(args.n_problems)})
        problems = [pall[i] for i in idx]
    else:
        problems = pall
    print(f"[load] {len(problems)} problems; ranking({args.rank_key}) over {len(ranked_layers)} layers; {time.time()-t0:.1f}s", flush=True)

    # ---- Phase 1: bf16 reference + per-layer weights -> CPU, then free ----
    print("[k] loading bf16 base (ref + weights) ...", flush=True)
    bf16 = AutoModelForCausalLM.from_pretrained(BF16_DIR, dtype=torch.bfloat16, trust_remote_code=True).to(DEVICE).eval()
    batch = build_batch(tok, problems, args.max_new_tokens, args.min_new_tokens, bf16)
    n_scored = sum(b["C"] for b in batch)
    print(f"[k] batch={len(batch)} scored={n_scored}", flush=True)
    ref_logp, ref_argmax = [], []
    for b in batch:
        lp, _ = teacher_forced_logprobs(bf16, b["seq"], b["scored"], want_hidden=False)
        ref_logp.append(lp)
        ref_argmax.append(lp.argmax(dim=-1))
    bf16_sd = [{k: v.detach().cpu() for k, v in layer_shared_sd(L).items()} for L in get_layers(bf16)]
    n_layers = len(bf16_sd)
    assert set(ranked_layers) == set(range(n_layers)), "ranking must cover all body layers"
    del bf16
    gc.collect(); torch.cuda.empty_cache()
    print(f"[k] n_layers={n_layers}; bf16 freed; {time.time()-t0:.1f}s", flush=True)

    # ---- Phase 2: int4-dense working model on GPU ----
    print("[k] loading int4 (run_compressed=False) ...", flush=True)
    qc = CompressedTensorsConfig(run_compressed=False)
    int4 = AutoModelForCausalLM.from_pretrained(
        INT4_ID, dtype=torch.bfloat16, trust_remote_code=True, quantization_config=qc
    ).to(DEVICE).eval()
    layers = get_layers(int4)

    @torch.inference_mode()
    def eval_div():
        kl_sum = 0.0; flips = 0; n = 0
        for i, b in enumerate(batch):
            lp, _ = teacher_forced_logprobs(int4, b["seq"], b["scored"], want_hidden=False)
            rlp = ref_logp[i].float(); p = rlp.exp()
            kl_sum += (p * (rlp - lp.float())).sum(dim=-1).sum().item()
            flips += (lp.argmax(dim=-1) != ref_argmax[i]).sum().item()
            n += lp.shape[0]
            del lp, rlp, p
        return kl_sum / n, flips / n

    ks = sorted({(n_layers if t.strip() == "all" else int(t)) for t in args.k_grid.split(",")})
    ks = [k for k in ks if 0 <= k <= n_layers]

    curve = []
    restored = set()
    d0 = None
    for k in ks:
        target = set(ranked_layers[:k])
        for li in sorted(target - restored):           # cumulative incremental restore
            layers[li].load_state_dict(bf16_sd[li], strict=False)
            restored.add(li)
        ts = time.time()
        kl, flip = eval_div()
        if k == 0:
            d0 = kl
        rec = (d0 - kl) / d0 if d0 else 0.0
        curve.append({"k": k, "kl": kl, "flip": flip, "recovery": rec,
                      "restored": sorted(restored), "wall_s": time.time() - ts})
        print(f"  [k={k:2d}] KL={kl:.5f} flip={flip:.4f} recovery={rec:+.4f} "
              f"(restored {len(restored)}/{n_layers}; {time.time()-ts:.0f}s)", flush=True)

    by_k = {c["k"]: c for c in curve}
    d_int4 = by_k[0]["kl"]
    d_full = by_k[n_layers]["kl"]
    # self-checks: k=42 should be ~full bf16 (KL ~ 0); recovery monotonic?
    full_bf16_clean = bool(d_full < 1e-4)
    recs = [c["recovery"] for c in curve]
    monotonic = all(recs[i] <= recs[i + 1] + 1e-9 for i in range(len(recs) - 1))
    n_for_90 = next((c["k"] for c in curve if c["recovery"] >= REC_BAR), ">all")
    # concentration verdict at the QUALITY-proxy (cumulative) level
    rec_at_conc_k = next((c["recovery"] for c in curve if c["k"] >= CONC_K), recs[-1])
    kl_collapse_concentrated = bool(isinstance(n_for_90, int) and n_for_90 <= CONC_K)

    result = {
        "analysis_only": True, "official_tps": 0, "metric": "kl_recovery",
        "rank_key": args.rank_key, "ranked_layers": ranked_layers,
        "n_problems": len(batch), "n_scored_tokens": n_scored, "n_layers": n_layers,
        "max_new_tokens": args.max_new_tokens,
        "D_int4_kl": d_int4, "D_full_bf16_kl": d_full, "full_bf16_clean": full_bf16_clean,
        "restore_k_kl": [[c["k"], c["kl"]] for c in curve],
        "restore_k_recovery": [[c["k"], c["recovery"]] for c in curve],
        "recovery_monotonic": monotonic,
        "conc_k": CONC_K, "rec_bar": REC_BAR,
        "recovery_at_conc_k": rec_at_conc_k,
        "n_layers_for_90pct_recovery": n_for_90,
        "kl_collapse_concentrated": kl_collapse_concentrated,
        "curve": curve,
        "elapsed_s": time.time() - t0,
    }
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in [
        "restore_k_recovery", "recovery_at_conc_k", "n_layers_for_90pct_recovery",
        "kl_collapse_concentrated", "recovery_monotonic", "full_bf16_clean"]}, indent=2), flush=True)
    print(f"[done] wrote {args.out}; {time.time()-t0:.1f}s", flush=True)

    if not args.no_wandb:
        import wandb
        run = wandb.init(entity="wandb-applied-ai-team", project="gemma-challenge-senpai",
                         name="stark/aime-quant-sensitivity-locate-restorek-kl",
                         group="aime-quant-sensitivity-locate",
                         config={"analysis_only": True, "official_tps": 0, "metric": "kl_recovery",
                                 "rank_key": args.rank_key, "n_problems": len(batch),
                                 "n_scored_tokens": n_scored})
        wandb.summary.update({k: result[k] for k in [
            "D_int4_kl", "D_full_bf16_kl", "recovery_at_conc_k", "n_layers_for_90pct_recovery",
            "kl_collapse_concentrated", "recovery_monotonic", "full_bf16_clean"]})
        tbl = wandb.Table(columns=["k", "kl", "flip", "recovery"])
        for c in curve:
            tbl.add_data(c["k"], c["kl"], c["flip"], c["recovery"])
            wandb.log({"k": c["k"], "restore_k_kl": c["kl"], "restore_k_recovery": c["recovery"]})
        wandb.log({"restore_k_kl_curve": tbl})
        run.finish(); print(f"[wandb] run id: {run.id}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
