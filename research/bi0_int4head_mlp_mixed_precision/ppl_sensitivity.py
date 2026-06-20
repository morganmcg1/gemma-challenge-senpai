#!/usr/bin/env python
"""PR #810 Step 1b — per-layer body-MLP PPL-delta sensitivity (W4 base, W3/W2 single-layer).

OFFLINE, no serving. Replicates the OFFICIAL challenge PPL exactly via a
teacher-forced forward over `ppl_ground_truth_tokens.jsonl` (128 records,
context+target). PPL = exp(mean NLL of target tokens given context), identical to
`speed_benchmark/scripts/ppl_endpoint.py` (which scores prompt_logprobs at the
target positions). prompt_logprob[i] = log P(tok_i | tok_<i) = log_softmax(logits[i-1])[tok_i].

Base = the served int4 body, loaded as bf16 (run_compressed=False -> W4deq Linear
weights). Per layer we FAKE-QUANT that one MLP (gate/up/down) to W3 (then W2)
group_size=32 symmetric (the same scheme the body uses), recompute PPL, and report
ΔPPL vs the all-W4 baseline. Also evaluates a few cumulative mixed configs on the
full 128 records to anchor the bytes-saved-vs-PPL Pareto front.

LOCAL ONLY. No HF Job, no submission.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # local A10G: only GPU 0 exists (inherited =1 is wrong)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from compressed_tensors.quantization import QuantizationArgs  # noqa: E402
from compressed_tensors.quantization.lifecycle.forward import quantize, dequantize  # noqa: E402
from compressed_tensors.quantization.utils.helpers import calculate_qparams  # noqa: E402

GROUP_SIZE = 32
N_LAYERS = 42
PROJS = ["gate_proj", "up_proj", "down_proj"]
DEFAULT_SRC = os.path.expanduser(
    "~/.cache/huggingface/hub/models--google--gemma-4-E4B-it-qat-w4a16-ct/"
    "snapshots/ef0a4c43726bde42a3ca04fd300397c0b8b3c3f0")
PPL_DATA = ("official/main_bucket/shared_resources/speed_benchmark/data/"
            "ppl_ground_truth_tokens.jsonl")


def make_qargs(num_bits: int) -> QuantizationArgs:
    return QuantizationArgs(num_bits=num_bits, type="int", strategy="group",
                            group_size=GROUP_SIZE, symmetric=True, observer="minmax")


def fakequant_g32_sym(w: torch.Tensor, num_bits: int) -> torch.Tensor:
    """Quantize->dequantize a weight at num_bits g32 symmetric. Returns bf16."""
    wf = w.to(torch.float32)
    out_dim, in_dim = wf.shape
    qargs = make_qargs(num_bits)
    ng = in_dim // GROUP_SIZE
    wg = wf.reshape(out_dim, ng, GROUP_SIZE)
    scale, zp = calculate_qparams(wg.amin(dim=-1), wg.amax(dim=-1), qargs)
    q = quantize(wf, scale, zp, qargs)
    deq = dequantize(q, scale, zp, qargs)
    return deq.to(torch.bfloat16)


def load_records(path: str):
    recs = []
    for ln in Path(path).read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        r = json.loads(ln)
        ctx = r["context_token_ids"]
        tgt = r["target_token_ids"]
        ids = ctx + tgt
        recs.append({"id": str(r.get("id", len(recs))), "ids": ids,
                     "score_start": len(ctx), "score_end": len(ids)})
    return recs


def get_layers(model):
    """Return the ModuleList of text decoder layers (robust to nesting)."""
    for path in ("model.language_model.layers", "model.model.language_model.layers",
                 "language_model.model.layers", "model.layers"):
        obj = model
        ok = True
        for part in path.split("."):
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                ok = False
                break
        if ok and hasattr(obj, "__len__") and len(obj) == N_LAYERS:
            return obj
    raise RuntimeError("could not locate text decoder layers")


@torch.no_grad()
def ppl_over(model, recs, device, max_records=None):
    total_nll = 0.0
    total_tok = 0
    n = 0
    for r in recs:
        if max_records is not None and n >= max_records:
            break
        ids = torch.tensor(r["ids"], dtype=torch.long, device=device)
        out = model(input_ids=ids.unsqueeze(0), use_cache=False)
        logits = out.logits[0]  # [T, V]
        s, e = r["score_start"], r["score_end"]
        pred = logits[s - 1:e - 1].float()        # predicts tokens s..e-1
        tgt = ids[s:e]
        nll = F.cross_entropy(pred, tgt, reduction="sum")
        total_nll += float(nll)
        total_tok += int(e - s)
        n += 1
        del out, logits, pred
    return math.exp(total_nll / total_tok), total_tok, n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--ppl-data", default=PPL_DATA)
    ap.add_argument("--subset", type=int, default=32,
                    help="records (stride-sampled) used for the per-layer sweep")
    ap.add_argument("--bits", default="3,2", help="single-layer down-quant bit-widths to sweep")
    ap.add_argument("--smoke", action="store_true", help="2 records, layers [0,21] only")
    ap.add_argument("--recon-json",
                    default="research/bi0_int4head_mlp_mixed_precision/recon_sensitivity.json")
    ap.add_argument("--output",
                    default="research/bi0_int4head_mlp_mixed_precision/ppl_sensitivity.json")
    ap.add_argument("--wandb_project", default=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"))
    ap.add_argument("--wandb_entity", default=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"))
    ap.add_argument("--wandb_group", default="body-mlp-mixed-precision")
    ap.add_argument("--wandb_name", default="stark/ppl-sensitivity-map")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    bits_sweep = [int(x) for x in args.bits.split(",") if x.strip()]
    device = torch.device("cuda:0")
    assert torch.cuda.is_available(), "no CUDA (check CUDA_VISIBLE_DEVICES=0)"
    print(f"[ppl] device={torch.cuda.get_device_name(0)} torch={torch.__version__}", flush=True)

    # ---- load model as bf16 (run_compressed=False -> dense W4deq Linear weights) ----
    t0 = time.time()
    from transformers import AutoModelForCausalLM
    try:
        from transformers import CompressedTensorsConfig
        qcfg = CompressedTensorsConfig(run_compressed=False)
    except Exception:  # noqa: BLE001
        qcfg = None
    print("[ppl] loading model (bf16, run_compressed=False) ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.src, dtype=torch.bfloat16,
        **({"quantization_config": qcfg} if qcfg is not None else {}))
    model.eval()
    # drop multimodal towers (unused for text PPL) to save VRAM
    dropped = []
    base = model.model if hasattr(model, "model") else model
    for tower in ("vision_tower", "audio_tower", "embed_vision", "embed_audio"):
        if hasattr(base, tower) and getattr(base, tower) is not None:
            setattr(base, tower, None)
            dropped.append(tower)
    model.to(device)
    torch.cuda.synchronize()
    print(f"[ppl] loaded in {time.time()-t0:.0f}s; dropped towers: {dropped}; "
          f"GPU mem {torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)

    layers = get_layers(model)
    # sanity: confirm a gate_proj weight is a plain bf16 dense tensor
    g0 = layers[0].mlp.gate_proj.weight
    print(f"[ppl] layer0 gate_proj.weight {tuple(g0.shape)} {g0.dtype} "
          f"(dense bf16 expected)", flush=True)

    # snapshot original (W4deq) MLP weights on CPU
    orig = {}
    for L in range(N_LAYERS):
        for p in PROJS:
            w = getattr(layers[L].mlp, p).weight
            orig[(L, p)] = w.detach().to("cpu").clone()

    def patch_layer(L, num_bits):
        for p in PROJS:
            w = orig[(L, p)].to(device)
            getattr(layers[L].mlp, p).weight.data.copy_(fakequant_g32_sym(w, num_bits))

    def restore_layer(L):
        for p in PROJS:
            getattr(layers[L].mlp, p).weight.data.copy_(orig[(L, p)].to(device))

    recs = load_records(args.ppl_data)
    # stride-sample a deterministic subset for the per-layer sweep
    stride = max(1, len(recs) // args.subset)
    subset = recs[::stride][:args.subset]
    if args.smoke:
        subset = recs[:2]
    sweep_layers = [0, 21] if args.smoke else list(range(N_LAYERS))
    print(f"[ppl] {len(recs)} records; sweep subset={len(subset)} (stride {stride}); "
          f"layers={len(sweep_layers)} bits={bits_sweep}", flush=True)

    run = None
    if not args.no_wandb:
        try:
            import wandb
            run = wandb.init(entity=args.wandb_entity, project=args.wandb_project,
                             group=args.wandb_group, name=args.wandb_name, job_type="analysis",
                             config={"group_size": GROUP_SIZE, "n_layers": N_LAYERS,
                                     "subset": len(subset), "bits": bits_sweep,
                                     "ppl_records": len(recs), "src": args.src,
                                     "dropped_towers": dropped})
            print(f"[ppl] W&B run: {run.url}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[ppl] W&B init failed: {exc!r}", flush=True)

    # ---- baselines ----
    base_sub_ppl, _, _ = ppl_over(model, subset, device)
    base_full_ppl, full_tok, full_n = ppl_over(model, recs, device)
    print(f"[ppl] BASELINE all-W4: subset_ppl={base_sub_ppl:.4f} "
          f"full_ppl={base_full_ppl:.4f} ({full_n} recs, {full_tok} tok)", flush=True)

    # ---- per-layer single-down sweep (on subset) ----
    sweep = []  # {layer,bits,ppl,delta_ppl}
    sweep_t0 = time.time()
    for L in sweep_layers:
        for b in bits_sweep:
            patch_layer(L, b)
            ppl, _, _ = ppl_over(model, subset, device)
            restore_layer(L)
            d = ppl - base_sub_ppl
            sweep.append({"layer": L, "bits": b, "ppl": ppl, "delta_ppl": d})
            if run is not None:
                run.log({f"W{b}_single_delta_ppl": d, f"W{b}_single_ppl": ppl, "layer_idx": L})
        if L % 6 == 0 or L == sweep_layers[-1]:
            print(f"[ppl] swept layer {L:2d} ({time.time()-sweep_t0:.0f}s)", flush=True)

    # rank layers by W3 single-layer ΔPPL (most robust = smallest delta)
    w3 = sorted([s for s in sweep if s["bits"] == 3], key=lambda s: s["delta_ppl"])
    print("\n[ppl] per-layer single-W3 ΔPPL (subset), most→least robust:", flush=True)
    for s in w3:
        print(f"  L{s['layer']:2d}: ΔPPL={s['delta_ppl']:+.4f} (ppl {s['ppl']:.4f})", flush=True)

    payload = {
        "config": {"group_size": GROUP_SIZE, "n_layers": N_LAYERS,
                   "subset_records": len(subset), "bits": bits_sweep,
                   "ppl_records": len(recs), "src": args.src, "dropped_towers": dropped},
        "baseline": {"subset_ppl": base_sub_ppl, "full_ppl": base_full_ppl,
                     "full_tokens": full_tok, "full_records": full_n},
        "single_layer_sweep": sweep,
    }

    # ---- cumulative mixed configs on FULL 128 (anchor the Pareto front) ----
    if not args.smoke:
        # robustness rank from W3 single-layer ΔPPL
        rank_w3 = [s["layer"] for s in w3]
        configs = {
            "all_W3": {L: 3 for L in range(N_LAYERS)},
            "robust21_W3": {L: 3 for L in rank_w3[:21]},      # most-robust half -> W3
            "robust10_W3": {L: 3 for L in rank_w3[:10]},
            "robust10_W2": {L: 2 for L in rank_w3[:10]},
        }
        cfg_results = {}
        for name, assign in configs.items():
            for L, b in assign.items():
                patch_layer(L, b)
            ppl, tok, nrec = ppl_over(model, recs, device)
            for L in assign:
                restore_layer(L)
            # bytes saved per token vs all-W4 (weights only; bf16 scales fixed)
            saved = sum((1 - {3: 0.75, 2: 0.5}[b]) for L, b in assign.items())  # layer-equivalents
            cfg_results[name] = {"full_ppl": ppl, "n_layers_down": len(assign),
                                 "layer_equiv_weight_bytes_saved": saved,
                                 "passes_cap_2.42": ppl <= 2.42}
            print(f"[ppl] config {name:14s}: full_ppl={ppl:.4f} "
                  f"({'PASS' if ppl<=2.42 else 'FAIL'} cap2.42; ΔPPL "
                  f"{ppl-base_full_ppl:+.4f}; {len(assign)} layers down)", flush=True)
            if run is not None:
                run.log({f"config_full_ppl/{name}": ppl})
        payload["cumulative_configs"] = cfg_results

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(payload, fh, indent=2)
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"[ppl] wrote {args.output}; peak GPU mem {peak:.2f}GB; total {time.time()-t0:.0f}s", flush=True)

    if run is not None:
        try:
            import wandb
            tbl = wandb.Table(columns=["layer", "bits", "ppl", "delta_ppl"])
            for s in sweep:
                tbl.add_data(s["layer"], s["bits"], s["ppl"], s["delta_ppl"])
            run.log({"ppl_single_layer_table": tbl})
            run.summary.update({"baseline_full_ppl": base_full_ppl,
                                "baseline_subset_ppl": base_sub_ppl,
                                "peak_gpu_mem_gib": peak})
            if not args.smoke:
                for name, r in payload["cumulative_configs"].items():
                    run.summary.update({f"cfg_{name}_full_ppl": r["full_ppl"]})
            run.finish()
        except Exception as exc:  # noqa: BLE001
            print(f"[ppl] W&B log failed: {exc!r}", flush=True)


if __name__ == "__main__":
    main()
