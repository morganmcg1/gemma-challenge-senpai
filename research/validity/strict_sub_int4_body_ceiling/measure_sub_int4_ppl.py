#!/usr/bin/env python
"""GPU-MEASURED sub-int4 body PPL for PR #356 (the make-or-break half).

The CPU sibling ``strict_sub_int4_body_ceiling.py`` sized WHICH body bit-widths
lift the strict TPS ceiling past 500 (b*=3.72 bpw; b=3 ceiling 585 TPS). It used
a Llama-2 literature PPL band. The advisor (issue #319, human "measure don't
guess") sent the card back for the decisive measurement: actually quantize the
Gemma-4-E4B BODY to b=3 and measure REAL PPL on the official eval set against the
binding <= 2.42 gate. This script does that.

Substrate
---------
Base = the deployed int4 substrate ``google/gemma-4-E4B-it-qat-w4a16-ct``
(compressed-tensors pack-quantized int4 W4A16, group_size 32 over the 343
language-model body Linears; bf16 lm_head / embed / norms / vision+audio towers).
Loaded with ``run_compressed=False`` so each body module is a DENSE bf16 Linear
whose weight == the dequantized int4 (the maskllm_2to4 #118 methodology, validated
there to reproduce served PPL within 0.17%). The QAT base is the deployed model and
its own offline PPL anchors the documented ~2.01.

What it measures
----------------
Body-only quant (the 343 modules in ``official_quantized_modules.json`` =
1.6973824 GB int4 body, the exact set the ceiling-model body-read is computed
over). lm_head / embed / norms / towers stay bf16 ("deployed precision",
per the ceiling model). For each (method, bits, group) it fake-quantizes the body
in place and runs the official teacher-forced PPL over the fixed 128-record /
61,797-token ground-truth span (``exp(sum_nll / sum_tok)``), then restores.

Methods
-------
* rtn  : compressed-tensors symmetric min-max group RTN -- the EXACT deployed
         quantizer (build_quant.py) at fewer bits. The "uniform b=3" number.
* hqq  : HQQ (Badri & Shaji) half-quadratic data-free quant -- a strong, named
         PTQ method; gives int3 its best practical shot with no calibration.
* mse  : asymmetric per-group MSE-optimal clip RTN -- dependency-free cross-check.

Verdict (deployed-anchored)
---------------------------
``ppl_at_best_sub_int4_bits`` (MEASURED) = best (min) measured raw b=3 PPL. But the
raw number lives on this substrate's OWN int4 baseline (~2.01, bf16 head + g32
body) -- NOT the deployed operating point. The deployed #52 frontier (osoi5: int4
g128 body + PCK04 int4 head) reads 2.3772 served == 2.3812 offline on this same
128-rec span (maskllm #118, "directly comparable to the 2.42 cap"), so the gate
budget is only +0.0428 abs / +1.80% rel ABOVE 2.3772. The b=3 lever's MEASURED cost
(delta vs this substrate's int4 baseline) is transferred onto 2.3772 both additively
and multiplicatively; ``sub_int4_clears_500_strict`` = (both transfers <= 2.42).
b=3 ceiling is 585 TPS > 500, so clearing the *deployed-anchored* PPL gate at b=3
would be clearing 500 strict -- the measurement decides whether it does.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from local_validation import paths  # noqa: E402

DEFAULT_CKPT = "google/gemma-4-E4B-it-qat-w4a16-ct"
BODY_LIST = ROOT / "submissions" / "int4_g128_lmhead" / "official_quantized_modules.json"

PPL_GATE = 2.42
DEPLOYED_FRONTIER_PPL = 2.3772      # #52 int4 g128 + PCK04 head frontier
QAT_BASE_DOC_PPL = 2.01            # documented int4 QAT base PPL (ppl_runner / int4_qat)
BODY_INT4_GB = 1.6973824           # int4 body weight bytes (ceiling-model body-read)
STRICT_CEILING_B3 = 585.04         # strict ceiling at b=3 (CPU sibling)
STRICT_CEILING_B4 = 473.5296       # strict ceiling at b=4 (int4)


# --------------------------------------------------------------------------- #
# model + body
# --------------------------------------------------------------------------- #
def load_model(ckpt: str):
    from transformers import Gemma4ForConditionalGeneration
    from transformers.utils.quantization_config import CompressedTensorsConfig

    t0 = time.time()
    model = Gemma4ForConditionalGeneration.from_pretrained(
        ckpt,
        dtype=torch.bfloat16,
        quantization_config=CompressedTensorsConfig(run_compressed=False),
        low_cpu_mem_usage=True,
        device_map="cuda:0",
    ).eval()
    print(f"[ppl] model loaded in {time.time()-t0:.1f}s "
          f"gpu={torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)
    return model


def discover_body(model) -> dict:
    """Return {name -> module} for the 343 int4 body Linears (dense bf16)."""
    body_names = set(json.load(open(BODY_LIST)))
    assert len(body_names) == 343, f"expected 343 body modules, got {len(body_names)}"
    found = {}
    for name, mod in model.named_modules():
        if name in body_names and getattr(mod, "weight", None) is not None and mod.weight.dim() == 2:
            found[name] = mod
    missing = body_names - set(found)
    if missing:
        raise RuntimeError(f"missing {len(missing)} body modules, e.g. {sorted(missing)[:5]}")
    return found


def load_records():
    ep = paths.import_ppl_endpoint()
    return [ep.normalized_record(r, i)
            for i, r in enumerate(ep.read_records(paths.ppl_dataset()))]


# --------------------------------------------------------------------------- #
# PPL (official teacher-forced formula, full-vocab head)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def compute_ppl(model, records, limit=None) -> dict:
    recs = records if limit is None else records[:limit]
    tot_nll = 0.0
    tot_tok = 0
    for rec in recs:
        ids = torch.tensor(rec["prompt_token_ids"], dtype=torch.long, device="cuda:0").unsqueeze(0)
        ss, se = rec["score_start"], rec["score_end"]
        logits = model(input_ids=ids).logits[0]
        lp = torch.log_softmax(logits[ss - 1:se - 1].float(), dim=-1)
        gt = ids[0, ss:se]
        tot_nll += -lp[torch.arange(se - ss, device="cuda:0"), gt].sum().item()
        tot_tok += (se - ss)
    return {"ppl": math.exp(tot_nll / tot_tok), "nll": tot_nll, "ntok": tot_tok}


# --------------------------------------------------------------------------- #
# quantizers (all return a dequantized bf16 weight of the same shape)
# --------------------------------------------------------------------------- #
def _eff_group(in_features: int, group: int) -> int:
    """Largest valid group <= requested that divides in_features (fallback per-row)."""
    if group <= 0 or in_features % group == 0:
        return group if group > 0 else in_features
    return in_features  # per-row channel-wise fallback


@torch.no_grad()
def fakequant_rtn_ct(w: torch.Tensor, bits: int, group: int) -> torch.Tensor:
    """Compressed-tensors symmetric min-max group RTN — the deployed quantizer."""
    from compressed_tensors.quantization import QuantizationArgs
    from compressed_tensors.quantization.lifecycle.forward import quantize, dequantize
    from compressed_tensors.quantization.utils.helpers import calculate_qparams

    orig_dtype = w.dtype
    wf = w.to(torch.float32)
    out, inn = wf.shape
    g = _eff_group(inn, group)
    if g == inn:
        qa = QuantizationArgs(num_bits=bits, type="int", strategy="channel",
                              symmetric=True, observer="minmax")
        mn = wf.amin(dim=-1, keepdim=True)
        mx = wf.amax(dim=-1, keepdim=True)
    else:
        qa = QuantizationArgs(num_bits=bits, type="int", strategy="group",
                              group_size=g, symmetric=True, observer="minmax")
        ng = inn // g
        wg = wf.reshape(out, ng, g)
        mn = wg.amin(dim=-1)
        mx = wg.amax(dim=-1)
    scale, zp = calculate_qparams(mn, mx, qa)
    q = quantize(wf, scale, zp, qa)
    deq = dequantize(q, scale, zp, qa)
    return deq.to(orig_dtype)


@torch.no_grad()
def fakequant_hqq(w: torch.Tensor, bits: int, group: int) -> torch.Tensor:
    """HQQ half-quadratic data-free quant (axis=1 -> groups along the in/K dim)."""
    from hqq.core.quantize import Quantizer

    orig_dtype = w.dtype
    inn = w.shape[1]
    g = _eff_group(inn, group)
    wq, meta = Quantizer.quantize(
        w.to(torch.float32), nbits=bits, group_size=g, axis=1,
        optimize=True, round_zero=(bits >= 4), bitpack=False, device="cuda:0",
    )
    deq = Quantizer.dequantize(wq, meta)
    return deq.to(orig_dtype).reshape(w.shape)


@torch.no_grad()
def fakequant_mse_asym(w: torch.Tensor, bits: int, group: int, n_grid: int = 40) -> torch.Tensor:
    """Asymmetric per-group RTN with an MSE-optimal symmetric clip search."""
    orig_dtype = w.dtype
    wf = w.to(torch.float32)
    out, inn = wf.shape
    g = _eff_group(inn, group)
    ng = inn // g
    wg = wf.reshape(out, ng, g)
    qmax = float(2 ** bits - 1)
    wmin = wg.amin(dim=-1, keepdim=True)
    wmax = wg.amax(dim=-1, keepdim=True)
    mid = 0.5 * (wmax + wmin)
    half = 0.5 * (wmax - wmin).clamp_min(1e-9)
    best_err = None
    best_deq = None
    for i in range(n_grid):
        r = 1.0 - 0.5 * (i / (n_grid - 1))  # shrink range 1.0 -> 0.5
        lo = mid - r * half
        hi = mid + r * half
        scale = (hi - lo) / qmax
        zp = torch.round(-lo / scale)
        q = torch.clamp(torch.round(wg / scale) + zp, 0, qmax)
        deq = (q - zp) * scale
        err = ((wg - deq) ** 2).sum(dim=-1, keepdim=True)
        if best_err is None:
            best_err, best_deq = err, deq
        else:
            take = err < best_err
            best_err = torch.where(take, err, best_err)
            best_deq = torch.where(take, deq, best_deq)
    return best_deq.reshape(out, inn).to(orig_dtype)


QUANTIZERS = {
    "rtn": fakequant_rtn_ct,
    "hqq": fakequant_hqq,
    "mse": fakequant_mse_asym,
}


# --------------------------------------------------------------------------- #
# apply / snapshot / restore
# --------------------------------------------------------------------------- #
@torch.no_grad()
def snapshot(body: dict) -> dict:
    return {n: m.weight.detach().to("cpu", copy=True) for n, m in body.items()}


@torch.no_grad()
def restore(body: dict, snap: dict):
    for n, m in body.items():
        m.weight.copy_(snap[n].to(m.weight.device))


@torch.no_grad()
def apply_quant(body: dict, method: str, bits: int, group: int) -> dict:
    fq = QUANTIZERS[method]
    rel_errs = []
    t0 = time.time()
    for m in body.values():
        w = m.weight
        deq = fq(w, bits, group)
        rel = (w.float() - deq.float()).norm() / w.float().norm().clamp_min(1e-9)
        rel_errs.append(float(rel))
        m.weight.copy_(deq)
    return {
        "n_modules": len(body),
        "rel_err_mean": sum(rel_errs) / len(rel_errs),
        "rel_err_max": max(rel_errs),
        "apply_s": time.time() - t0,
    }


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def parse_configs(spec: str) -> list:
    """spec: 'method:bits:group,method:bits:group,...'"""
    out = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        method, bits, group = item.split(":")
        out.append({"method": method, "bits": int(bits), "group": int(group)})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--limit", type=int, default=None, help="score first N records (smoke)")
    ap.add_argument("--configs", default=(
        # b=3 at the deployed body grouping (g128) AND the finest/most-favorable
        # grouping (g32), three methods each, + the b=2 probe. The g128 rows are
        # the deployment-faithful number (the deployed body is int4 g128); the g32
        # rows are the best-case lower bound on the b=3 cost.
        "rtn:3:128,rtn:3:32,mse:3:128,mse:3:32,hqq:3:128,hqq:3:64,hqq:3:32,hqq:2:32"
    ), help="comma list of method:bits:group")
    ap.add_argument("--out", default=str(ROOT / "research" / "validity" /
                                         "strict_sub_int4_body_ceiling" /
                                         "measure_sub_int4_ppl_results.json"))
    ap.add_argument("--wandb_project", default="senpai")
    ap.add_argument("--wandb_entity", default=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"))
    ap.add_argument("--wandb_group", default="strict-sub-int4-body-ceiling")
    ap.add_argument("--wandb_name", default="denken/strict-sub-int4-body-ceiling-ppl")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    for note in paths.prepare_local_gpu_env():
        print(f"[ppl] {note}", flush=True)

    configs = parse_configs(args.configs)
    model = load_model(args.checkpoint)
    body = discover_body(model)
    records = load_records()
    print(f"[ppl] body modules={len(body)} | records={len(records)} | "
          f"target_tokens={sum(r['score_end']-r['score_start'] for r in records)}", flush=True)

    # ---- baseline (b=4 anchor: the QAT model as loaded == dequant int4 g32) ---
    t0 = time.time()
    base = compute_ppl(model, records, limit=args.limit)
    print(f"[ppl] BASELINE (b=4 int4 QAT body) ppl={base['ppl']:.4f} "
          f"ntok={base['ntok']} ({time.time()-t0:.1f}s)  [doc int4 QAT ~{QAT_BASE_DOC_PPL}]", flush=True)

    snap = snapshot(body)
    results = []
    for cfg in configs:
        method, bits, group = cfg["method"], cfg["bits"], cfg["group"]
        qstats = apply_quant(body, method, bits, group)
        t0 = time.time()
        res = compute_ppl(model, records, limit=args.limit)
        restore(body, snap)
        row = {
            "method": method, "bits": bits, "group": group,
            "ppl": res["ppl"], "nll": res["nll"], "ntok": res["ntok"],
            "delta_vs_base": res["ppl"] - base["ppl"],
            "clears_gate": bool(res["ppl"] <= PPL_GATE),
            "rel_err_mean": qstats["rel_err_mean"], "rel_err_max": qstats["rel_err_max"],
            "quant_s": qstats["apply_s"], "ppl_s": time.time() - t0,
        }
        results.append(row)
        print(f"[ppl] {method} b{bits} g{group}: ppl={res['ppl']:.4f} "
              f"(Δ{res['ppl']-base['ppl']:+.4f}, rel_err {qstats['rel_err_mean']:.3f}) "
              f"{'PASS<=2.42' if row['clears_gate'] else 'OVER GATE'}  "
              f"[{qstats['apply_s']:.0f}s quant + {row['ppl_s']:.0f}s ppl]", flush=True)

    # ---- verdict (deployed-anchored, not raw-vs-gate) ----------------------- #
    # The raw b=3 PPL is measured on a substrate whose baseline (bf16 full head +
    # int4 g32 body) reads ~2.01 -- it is NOT the deployed operating point. The
    # deployed frontier (#52 osoi5, int4 g128 body + PCK04 int4 head) reads 2.3772
    # served == 2.3812 offline on THIS exact 128-rec span (maskllm #118, "directly
    # comparable to the 2.42 cap"). The 2.42 gate therefore has only +0.0428 abs /
    # +1.80% rel headroom above the deployed 2.3772. The b=3 lever must be charged
    # against THAT operating point: take the MEASURED b=3 body cost (delta vs this
    # substrate's own int4 baseline) and transfer it onto 2.3772, additively AND
    # multiplicatively. The lever clears 500-strict only if BOTH transfers stay
    # <= 2.42 (a config clears only if it survives either reading of the anchor).
    base_ppl = base["ppl"]
    int3 = [r for r in results if r["bits"] == 3]
    best_int3 = min(int3, key=lambda r: r["ppl"]) if int3 else None
    ppl_at_best_sub_int4_bits = best_int3["ppl"] if best_int3 else float("nan")

    def _anchor(ppl_raw: float) -> dict:
        d_abs = ppl_raw - base_ppl
        d_rel = d_abs / base_ppl
        add = DEPLOYED_FRONTIER_PPL + d_abs
        mul = DEPLOYED_FRONTIER_PPL * (ppl_raw / base_ppl)
        return {
            "raw_ppl": ppl_raw, "delta_abs_vs_substrate": d_abs,
            "delta_rel_vs_substrate": d_rel,
            "deployed_anchored_additive": add,
            "deployed_anchored_multiplicative": mul,
            "clears_gate_additive": bool(add <= PPL_GATE),
            "clears_gate_multiplicative": bool(mul <= PPL_GATE),
        }

    gate_headroom_abs = PPL_GATE - DEPLOYED_FRONTIER_PPL
    gate_headroom_rel = gate_headroom_abs / DEPLOYED_FRONTIER_PPL
    best_anchor = _anchor(ppl_at_best_sub_int4_bits) if best_int3 else None
    # deployment-faithful view: best b=3 at the deployed body grouping (g128).
    int3_g128 = [r for r in int3 if r["group"] == 128]
    best_int3_g128 = min(int3_g128, key=lambda r: r["ppl"]) if int3_g128 else None
    g128_anchor = _anchor(best_int3_g128["ppl"]) if best_int3_g128 else None
    sub_int4_clears_500_strict = bool(
        best_anchor
        and best_anchor["clears_gate_additive"]
        and best_anchor["clears_gate_multiplicative"]
    )

    payload = {
        "pr": 356,
        "agent": "denken",
        "leg": "GPU-measured sub-int4 body PPL (make-or-break half)",
        "checkpoint": args.checkpoint,
        "substrate_note": (
            "deployed int4 QAT substrate (google/gemma-4-E4B-it-qat-w4a16-ct), "
            "run_compressed=False dense bf16 body == dequant int4 g32; body-only "
            "quant of the 343 official body modules; lm_head/embed/norms/towers bf16."
        ),
        "ppl_gate": PPL_GATE,
        "deployed_frontier_ppl": DEPLOYED_FRONTIER_PPL,
        "deployed_anchor_note": (
            "deployed #52 osoi5 (int4 g128 body + PCK04 int4 head) reads 2.3772 "
            "served == 2.3812 offline on this 128-rec span (maskllm #118), directly "
            "comparable to the 2.42 cap; gate headroom above deployed is the budget."
        ),
        "gate_headroom_abs": gate_headroom_abs,
        "gate_headroom_rel": gate_headroom_rel,
        "qat_base_doc_ppl": QAT_BASE_DOC_PPL,
        "baseline_ppl_measured": base["ppl"],
        "baseline_ntok": base["ntok"],
        "baseline_reproduces_doc": bool(abs(base["ppl"] - QAT_BASE_DOC_PPL) < 0.06),
        "configs": results,
        "strict_ceiling_b3_tps": STRICT_CEILING_B3,
        "strict_ceiling_b4_tps": STRICT_CEILING_B4,
        "ppl_at_best_sub_int4_bits": ppl_at_best_sub_int4_bits,
        "best_int3_config": best_int3,
        "best_int3_deployed_anchored": best_anchor,
        "best_int3_g128_config": best_int3_g128,
        "best_int3_g128_deployed_anchored": g128_anchor,
        "sub_int4_clears_500_strict": sub_int4_clears_500_strict,
        "verdict_note": (
            f"best measured b=3 body PPL = {ppl_at_best_sub_int4_bits:.4f} (raw, on the "
            f"~{base_ppl:.4f} bf16-head/g32 substrate); its MEASURED cost vs this "
            f"substrate's int4 baseline is +{best_anchor['delta_abs_vs_substrate']:.4f} abs / "
            f"+{100*best_anchor['delta_rel_vs_substrate']:.2f}% rel. Charged onto the deployed "
            f"2.3772 anchor: additive={best_anchor['deployed_anchored_additive']:.4f}, "
            f"multiplicative={best_anchor['deployed_anchored_multiplicative']:.4f} -- both vs "
            f"gate {PPL_GATE} (headroom only +{gate_headroom_abs:.4f} / "
            f"+{100*gate_headroom_rel:.2f}%). clears_500_strict={sub_int4_clears_500_strict}."
        ),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[ppl] wrote {args.out}")
    print(f"[ppl] VERDICT: ppl_at_best_sub_int4_bits={ppl_at_best_sub_int4_bits:.4f} (raw) | "
          f"deployed-anchored add={best_anchor['deployed_anchored_additive']:.4f} "
          f"mul={best_anchor['deployed_anchored_multiplicative']:.4f} vs gate {PPL_GATE} "
          f"(deployed {DEPLOYED_FRONTIER_PPL}, headroom +{gate_headroom_abs:.4f}) | "
          f"sub_int4_clears_500_strict={sub_int4_clears_500_strict}", flush=True)

    if not args.no_wandb:
        try:
            _log_wandb(args, payload, base, results)
            with open(args.out, "w") as f:  # re-dump so the JSON captures run id
                json.dump(payload, f, indent=2)
        except Exception as exc:  # noqa: BLE001
            print(f"[ppl] W&B logging failed (non-fatal): {exc!r}", flush=True)

    return payload


def _log_wandb(args, payload, base, results):
    import wandb
    run = wandb.init(
        entity=args.wandb_entity, project=args.wandb_project,
        group=args.wandb_group, name=args.wandb_name, job_type="ppl-measure",
        config={
            "pr": 356, "agent": "denken", "checkpoint": args.checkpoint,
            "ppl_gate": PPL_GATE, "qat_base_doc_ppl": QAT_BASE_DOC_PPL,
            "deployed_frontier_ppl": DEPLOYED_FRONTIER_PPL,
            "n_body_modules": 343, "body_int4_gb": BODY_INT4_GB,
        },
    )
    tbl = wandb.Table(columns=["method", "bits", "group", "ppl", "delta_vs_base",
                               "rel_err_mean", "clears_gate"])
    for r in results:
        tbl.add_data(r["method"], r["bits"], r["group"], r["ppl"],
                     r["delta_vs_base"], r["rel_err_mean"], r["clears_gate"])
    run.log({"sub_int4_ppl_table": tbl})
    ba = payload["best_int3_deployed_anchored"]
    run.summary.update({
        "baseline_ppl_measured": payload["baseline_ppl_measured"],
        "baseline_reproduces_doc": payload["baseline_reproduces_doc"],
        "ppl_at_best_sub_int4_bits": payload["ppl_at_best_sub_int4_bits"],
        "best_int3_delta_rel_vs_substrate": ba["delta_rel_vs_substrate"],
        "deployed_anchored_additive": ba["deployed_anchored_additive"],
        "deployed_anchored_multiplicative": ba["deployed_anchored_multiplicative"],
        "gate_headroom_abs": payload["gate_headroom_abs"],
        "gate_headroom_rel": payload["gate_headroom_rel"],
        "sub_int4_clears_500_strict": payload["sub_int4_clears_500_strict"],
        "ppl_gate": PPL_GATE,
        "deployed_frontier_ppl": DEPLOYED_FRONTIER_PPL,
        "strict_ceiling_b3_tps": STRICT_CEILING_B3,
    })
    for r in results:
        tag = f"{r['method']}_b{r['bits']}_g{r['group']}"
        run.summary[f"ppl/{tag}"] = r["ppl"]
    run.finish()
    print(f"[ppl] W&B run: {run.url}", flush=True)
    payload["wandb_run_id"] = run.id


if __name__ == "__main__":
    main()
