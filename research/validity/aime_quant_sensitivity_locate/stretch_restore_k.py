#!/usr/bin/env python
"""STRETCH: restore-k AIME recovery curve (#586).

DIAGNOSTIC ONLY (analysis_only=true, official_tps=0). Ranks decoder layers by the
CLEAN quantize-from-bf16 sensitivity (quantize_result.json: ranked_layers_by_d_kl,
all-positive, un-confounded by QAT co-adaptation; --rank-key/--ranking override),
then restores the top-k to bf16 in the int4-dense working model and re-measures
**AIME greedy pass@1** (temperature=0, min_new_tokens=8 EOS-guard per #541) for
k in {0,2,5,10,all-body=42}. The recovery-curve SHAPE confirms concentrated vs
diffuse: a sharp jump at small k => concentrated; a gradual rise => diffuse.
k=0 is the full int4 floor; k=42 is the fully-bf16 (= unquantized) model and
defines 100% recovery, so the curve self-calibrates between my measured endpoints.

Note: greedy generation in dense-bf16 dequant (not Marlin); same token cap for all
k, so the curve is internally consistent. Absolute pass rates are anchored to the
cited int4 0.1167 / unquantized 0.400 but measured here under a fixed cap.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, CompressedTensorsConfig

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from research.downstream_quality_aime.aime_eval import (  # noqa: E402
    load_aime, build_messages, extract_answer,
)

BF16_DIR = "/senpai-run/home/student-ubel/.cache/huggingface/hub/models--google--gemma-4-E4B-it/snapshots/fee6332c1abaafb77f6f9624236c63aa2f1d0187"
INT4_ID = "google/gemma-4-E4B-it-qat-w4a16-ct"
DEVICE = "cuda:0"
GAP_LO, GAP_HI = 0.1167, 0.400   # cited int4 floor / unquantized anchors


def get_layers(model):
    return model.model.language_model.layers


@torch.inference_mode()
def aime_greedy_passrate(model, tokenizer, problems, max_new_tokens, min_new_tokens):
    n_correct = 0
    per = []
    for prob in problems:
        msgs = build_messages(prob["problem"])
        enc = tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True,
            enable_thinking=True, return_tensors="pt", return_dict=True,
        ).to(DEVICE)
        P = enc["input_ids"].shape[1]
        out = model.generate(
            **enc, max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens,
            do_sample=False, num_beams=1,
            pad_token_id=(tokenizer.pad_token_id or tokenizer.eos_token_id),
        )
        comp = out[0, P:]
        text = tokenizer.decode(comp, skip_special_tokens=True)
        ans = extract_answer(text)
        ok = (ans is not None and ans == prob["answer"])
        n_correct += int(ok)
        per.append({"id": prob["id"], "gold": prob["answer"], "ans": ans, "ok": ok,
                    "n_tok": int(comp.shape[0])})
    return n_correct / len(problems), per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="2024,2025")
    ap.add_argument("--n-problems", type=int, default=24)
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--min-new-tokens", type=int, default=8)
    ap.add_argument("--k-list", default="0,2,5,10,all")
    ap.add_argument("--ranking", default=str(HERE / "quantize_result.json"))
    ap.add_argument("--rank-key", default="ranked_layers_by_d_kl",
                    help="JSON key holding [[layer, score], ...] descending; "
                         "use ranked_layers_by_d_kl (clean quantize dir) or ranked_layers_by_s_kl (restore dir)")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--out", default=str(HERE / "restore_k_result.json"))
    args = ap.parse_args()

    t0 = time.time()
    rank = json.loads(Path(args.ranking).read_text())[args.rank_key]
    ranked_layers = [int(li) for li, _ in rank]   # descending sensitivity
    n_layers = len(ranked_layers)

    tokenizer = AutoTokenizer.from_pretrained(INT4_ID, trust_remote_code=True)
    problems_all = load_aime([y.strip() for y in args.years.split(",")], limit=None)
    if args.n_problems and args.n_problems < len(problems_all):
        stride = len(problems_all) / args.n_problems
        idx = sorted({int(i * stride) for i in range(args.n_problems)})
        problems = [problems_all[i] for i in idx]
    else:
        problems = problems_all
    print(f"[load] {len(problems)} problems; ranking over {n_layers} layers; {time.time()-t0:.1f}s", flush=True)

    # bf16 per-layer weights -> CPU, then free bf16 model
    print("[stretch] loading bf16 base to extract layer weights ...", flush=True)
    bf16 = AutoModelForCausalLM.from_pretrained(BF16_DIR, dtype=torch.bfloat16, trust_remote_code=True)
    bf16_sd = [{k: v.detach().cpu() for k, v in L.state_dict().items()} for L in get_layers(bf16)]
    del bf16
    import gc; gc.collect(); torch.cuda.empty_cache()

    # int4-dense working model on GPU
    print("[stretch] loading int4 (run_compressed=False) ...", flush=True)
    qc = CompressedTensorsConfig(run_compressed=False)
    int4 = AutoModelForCausalLM.from_pretrained(
        INT4_ID, dtype=torch.bfloat16, trust_remote_code=True, quantization_config=qc
    ).to(DEVICE).eval()
    layers = get_layers(int4)

    # parse k-list
    ks = []
    for tok in args.k_list.split(","):
        tok = tok.strip()
        ks.append(n_layers if tok == "all" else int(tok))
    ks = sorted(set(ks))

    curve = []
    restored = set()
    for k in ks:
        target = set(ranked_layers[:k]) if k < n_layers else set(range(n_layers))
        for li in sorted(target - restored):           # incremental restore
            layers[li].load_state_dict(bf16_sd[li], strict=False)
            restored.add(li)
        ts = time.time()
        pr, per = aime_greedy_passrate(int4, tokenizer, problems, args.max_new_tokens, args.min_new_tokens)
        curve.append({"k": k, "pass_rate": pr, "n_correct": int(round(pr * len(problems))),
                      "n_problems": len(problems), "wall_s": time.time() - ts})
        print(f"  [k={k:2d}] pass@1={pr:.4f} ({curve[-1]['n_correct']}/{len(problems)}) "
              f"restored={len(restored)} ({time.time()-ts:.0f}s)", flush=True)

    by_k = {c["k"]: c["pass_rate"] for c in curve}
    p0 = by_k.get(0, by_k[min(by_k)])
    pall = by_k.get(n_layers, by_k[max(by_k)])
    denom = (pall - p0)
    n_for_90 = ">all"
    if denom > 0:
        for c in curve:
            if (c["pass_rate"] - p0) / denom >= 0.90:
                n_for_90 = c["k"]
                break
    result = {
        "analysis_only": True, "official_tps": 0,
        "n_problems": len(problems), "max_new_tokens": args.max_new_tokens,
        "rank_key": args.rank_key,
        "ranked_layers": ranked_layers,
        "restore_k_aime": [[c["k"], c["pass_rate"]] for c in curve],
        "curve": curve,
        "p_k0": p0, "p_kall": pall,
        "n_layers_for_90pct_recovery": n_for_90,
        "cited_anchors": {"int4_floor": GAP_LO, "unquantized": GAP_HI, "bar_90pct": 0.360},
        "elapsed_s": time.time() - t0,
    }
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ["restore_k_aime", "p_k0", "p_kall", "n_layers_for_90pct_recovery"]}, indent=2), flush=True)
    print(f"[done] wrote {args.out}; {time.time()-t0:.1f}s", flush=True)

    if not args.no_wandb:
        import wandb
        run = wandb.init(entity="wandb-applied-ai-team", project="gemma-challenge-senpai",
                         name="stark/aime-quant-sensitivity-locate-restorek",
                         group="aime-quant-sensitivity-locate",
                         config={"analysis_only": True, "official_tps": 0,
                                 "n_problems": len(problems), "max_new_tokens": args.max_new_tokens})
        tbl = wandb.Table(columns=["k", "pass_rate", "n_correct"])
        for c in curve:
            tbl.add_data(c["k"], c["pass_rate"], c["n_correct"])
            wandb.log({"k": c["k"], "restore_k_pass_rate": c["pass_rate"]})
        wandb.summary.update({"p_k0": p0, "p_kall": pall, "n_layers_for_90pct_recovery": str(n_for_90)})
        wandb.log({"restore_k_curve": tbl})
        run.finish()
        print(f"[wandb] run id: {run.id}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
