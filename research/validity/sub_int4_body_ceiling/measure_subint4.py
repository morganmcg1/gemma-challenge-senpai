#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""GPU-backed sub-int4 body-quant MEASUREMENT (PR #355 pivot, lane = NON-spec).

The co-advisor pivoted PR #355 from a 0-GPU analytic card to a GPU-backed
measurement on the pod A10G ("if our GPUs accelerate work, use them"). This file
is the measured counterpart of the analytic `sub_int4_body_ceiling.py` card: the
card supplies the PREDICTION (a roofline BW model of tps(bits)); this file
supplies the MEASURED PPL + greedy-token-identity at each body bit-width.

WHAT IS MEASURED (offline transformers, CUDA, NO HF Job, NO submission)
----------------------------------------------------------------------
  * real PPL at body bit-widths {16 (bf16 ref), 4, 3, 2} on the OFFICIAL eval
    set `ppl_ground_truth_tokens.jsonl` (128 records), replicating
    `ppl_endpoint.py` EXACTLY: PPL = exp(sum NLL / sum scored tokens), the
    scored positions are the target tokens, each scored by log P(tok | prefix)
    (== vLLM prompt_logprobs=1, add_special_tokens=false). Binding gate: 2.42.
  * strict GREEDY-TOKEN-IDENTITY: greedy-decode K eval prompts at each width and
    measure token-divergence vs the int4 (b=4) baseline. Offline-vs-offline on
    one code path isolates the bit-width effect (BASELINE.md: a SERVED ref is
    needed only for the cross-stack #192 check; here we isolate the bit delta).
  * predicted single-stream TPS at each width = the analytic BW model imported
    from `sub_int4_body_ceiling.strict_nonspec_tps`. There is NO sub-4-bit
    weight kernel in vLLM 0.22 / compressed-tensors 0.15 (Marlin supports only
    [4,8] bits), so real sub-int4 latency is NOT directly measurable; fake-quant
    dequantizes back to bf16 so its forward time is bf16-bound and
    WIDTH-INVARIANT. Per the advisor these are relative headroom screens (local
    A10G != official a10g absolute scale, land #245 ~7x gap).

QUANT (the measured perturbation)
---------------------------------
Per-group (g128) RTN fake-quant of the TEXT-DECODER body linears
(self_attn.{q,k,v,o}_proj, mlp.{gate,up,down}_proj), dequantized to bf16.
lm_head + embeddings (tied) kept at source precision. Two schemes:
  * asym (default): asymmetric zero-point, 2**n levels (matches GPTQ/AWQ grid,
    uses all 4 levels at 2-bit -- avoids the symmetric ternary collapse).
  * sym: symmetric, 2**(n-1)-1 levels (matches the deployed compressed-tensors
    w4a16 symmetric grid; 2-bit degenerates to ternary).
RTN is the PESSIMISTIC bound: a real sub-int4 build would use GPTQ/AWQ/QAT,
strictly better. The card's literature curve (AQLM/QTIP) is the OPTIMISTIC
bound. The deployable-today truth (scalar Marlin) sits between.

CALIBRATION: the served int4 (w4a16-ct Marlin, QAT) measures PPL 2.0188 on this
exact dataset (research/greedy_ref_keying/int4_full/ppl_summary.json). Our
offline RTN-int4 on the QAT-unquantized bf16 base is a DIFFERENT int4 (RTN, not
Marlin-QAT), so it calibrates NEAR but not equal to 2.0188; the decision-grade
quantities are the DELTAS 4->3->2 and the identity break, not the absolute base.

GO CONDITION (the advisor's frontier question)
----------------------------------------------
Does ANY sub-int4 width hold PPL <= 2.42 AND greedy-identity (vs int4) while its
predicted TPS > 165.44 (lawine #196 non-spec frontier)? Reported per width and
overall. Lane split: denken #356 owns the SPEC-ceiling (473.5 cap); this leg
owns the NON-SPEC 165.44 AR frontier -- cross-referenced, not duplicated.

Run (full):
    cd target/ && CUDA_VISIBLE_DEVICES=0 WANDB_MODE=online .venv/bin/python \
      research/validity/sub_int4_body_ceiling/measure_subint4.py \
      --bits 16,4,3,2 --scheme asym \
      --wandb_group sub-int4-body-355 --wandb_name lawine/sub-int4-body-355-measured
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
TARGET_ROOT = HERE.parents[2]  # research/ is under target/

# Import the analytic card (prediction overlay) — same directory.
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import sub_int4_body_ceiling as card  # noqa: E402

# Anchors imported from the card / measured baselines (single source of truth).
STRICT_NONSPEC_FLOOR_TPS = card.STRICT_NONSPEC_FLOOR_TPS  # 165.44 (lawine #196)
PPL_GATE = card.PPL_GATE                                  # 2.42
TARGET_TPS = card.TARGET_TPS                              # 500.0
# Measured served-int4 PPL on THIS dataset (calibration anchor; non-spec frontier base).
SERVED_INT4_PPL_NONSPEC = 2.0188   # research/greedy_ref_keying/int4_full/ppl_summary.json
SERVED_INT4_PPL_SPEC = 2.3772      # deployed spec flagship PPL (the gate's named anchor)

DEFAULT_PPL_DATASET = (
    TARGET_ROOT
    / "official/main_bucket/shared_resources/speed_benchmark/data/ppl_ground_truth_tokens.jsonl"
)
DEFAULT_BASE_MODEL = "google/gemma-4-E4B-it-qat-q4_0-unquantized"

BODY_PROJ_SUFFIXES = (
    "q_proj", "k_proj", "v_proj", "o_proj",  # attention
    "gate_proj", "up_proj", "down_proj",      # mlp
)


# --------------------------------------------------------------------------- #
# Per-group RTN fake-quant.
# --------------------------------------------------------------------------- #
def fake_quant_per_group(w: torch.Tensor, n_bits: int, group_size: int,
                         scheme: str) -> torch.Tensor:
    """Round-to-nearest per-group fake-quant along the INPUT (contraction) dim.

    w: [out_features, in_features]. Groups of `group_size` input channels share a
    scale (and zero-point for asym). Returns a dequantized tensor of w.dtype.
    """
    if n_bits >= 16:
        return w  # bf16 reference: no quant.
    out_f, in_f = w.shape
    if in_f % group_size != 0:
        raise ValueError(f"in_features {in_f} not divisible by group_size {group_size}")
    wg = w.reshape(out_f, in_f // group_size, group_size).float()

    if scheme == "sym":
        qmax = float(2 ** (n_bits - 1) - 1)          # 4->7, 3->3, 2->1 (ternary)
        absmax = wg.abs().amax(dim=-1, keepdim=True)
        scale = (absmax / qmax).clamp(min=1e-8)
        q = torch.clamp(torch.round(wg / scale), -qmax, qmax)
        deq = q * scale
    elif scheme == "asym":
        qmax = float(2 ** n_bits - 1)                # 4->15, 3->7, 2->3 (full 2**n levels)
        wmin = wg.amin(dim=-1, keepdim=True)
        wmax = wg.amax(dim=-1, keepdim=True)
        scale = ((wmax - wmin) / qmax).clamp(min=1e-8)
        zp = torch.round(-wmin / scale)
        q = torch.clamp(torch.round(wg / scale) + zp, 0.0, qmax)
        deq = (q - zp) * scale
    else:
        raise ValueError(f"unknown scheme {scheme!r}")

    return deq.reshape(out_f, in_f).to(w.dtype)


def is_body_linear(name: str, module: torch.nn.Module) -> bool:
    """A text-decoder body projection: an nn.Linear under language_model.*.layers.*
    ending in a known proj suffix. Excludes lm_head, embeddings, vision/audio."""
    if not isinstance(module, torch.nn.Linear):
        return False
    if "language_model" not in name or ".layers." not in name:
        return False
    return name.endswith(BODY_PROJ_SUFFIXES)


# --------------------------------------------------------------------------- #
# Model load + body-weight snapshot.
# --------------------------------------------------------------------------- #
def load_model(base_model: str, device: str):
    from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

    tok = AutoTokenizer.from_pretrained(base_model)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        base_model, dtype=torch.bfloat16, low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    return model, tok


def snapshot_body(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    snap: dict[str, torch.Tensor] = {}
    for name, mod in model.named_modules():
        if is_body_linear(name, mod):
            snap[name] = mod.weight.detach().to("cpu", copy=True)
    return snap


def apply_width(model: torch.nn.Module, snap: dict[str, torch.Tensor],
                n_bits: int, group_size: int, scheme: str, device: str) -> dict[str, Any]:
    """Overwrite every body linear with its fake-quant of the ORIGINAL weight."""
    mods = dict(model.named_modules())
    n_layers = 0
    n_params = 0
    with torch.no_grad():
        for name, w0_cpu in snap.items():
            w0 = w0_cpu.to(device)
            wq = fake_quant_per_group(w0, n_bits, group_size, scheme)
            mods[name].weight.data.copy_(wq)
            n_layers += 1
            n_params += w0.numel()
            del w0, wq
    torch.cuda.synchronize() if device.startswith("cuda") else None
    return {"n_body_linears": n_layers, "n_body_params": n_params}


# --------------------------------------------------------------------------- #
# PPL (replicates official ppl_endpoint.py methodology).
# --------------------------------------------------------------------------- #
def read_ppl_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text().strip()
    if text and text[0] == "[":
        raw = json.loads(text)
    else:
        raw = [json.loads(ln) for ln in text.splitlines() if ln.strip()]
    out = []
    for i, r in enumerate(raw):
        if "context_token_ids" in r and "target_token_ids" in r:
            ctx, tgt = r["context_token_ids"], r["target_token_ids"]
            ids = ctx + tgt
            score_start, score_end = len(ctx), len(ids)
        elif "prompt_token_ids" in r:
            ids = r["prompt_token_ids"]
            score_start = int(r.get("score_token_start", r.get("score_start", 1)))
            score_end = int(r.get("score_token_end", len(ids)))
        else:
            raise ValueError(f"record {i} missing token fields")
        score_start = max(score_start, 1)
        out.append({"id": str(r.get("id", i)), "ids": ids,
                    "score_start": score_start, "score_end": score_end})
    return out


@torch.no_grad()
def measure_ppl(model, records: list[dict[str, Any]], device: str,
                max_records: int | None) -> dict[str, Any]:
    """exp(sum NLL / sum scored tokens). nll(idx) = -log P(ids[idx] | ids[:idx])
    = -log_softmax(logits[idx-1])[ids[idx]]  (HF logits[t] predicts token t+1)."""
    total_nll = 0.0
    total_tok = 0
    per_record = []
    recs = records if max_records is None else records[:max_records]
    for r in recs:
        ids = torch.tensor([r["ids"]], dtype=torch.long, device=device)
        logits = model(input_ids=ids, use_cache=False).logits[0].float()  # [seq, vocab]
        ss, se = r["score_start"], r["score_end"]
        # predict positions [ss, se) from logits at [ss-1, se-1)
        pred = torch.log_softmax(logits[ss - 1: se - 1], dim=-1)           # [n, vocab]
        tgt = ids[0, ss:se]                                                # [n]
        nll = -pred.gather(1, tgt.unsqueeze(1)).squeeze(1).sum().item()
        ntok = int(se - ss)
        total_nll += nll
        total_tok += ntok
        per_record.append({"id": r["id"], "nll": nll, "num_tokens": ntok,
                           "ppl": math.exp(nll / ntok)})
        del logits, pred
    return {
        "ppl": math.exp(total_nll / total_tok),
        "mean_record_ppl": sum(p["ppl"] for p in per_record) / len(per_record),
        "num_records": len(per_record),
        "num_tokens": total_tok,
        "neg_log_likelihood": total_nll,
        "per_record": per_record,
    }


# --------------------------------------------------------------------------- #
# Greedy-token-identity.
# --------------------------------------------------------------------------- #
@torch.no_grad()
def greedy_decode(model, prompt_ids: list[int], n_new: int, device: str) -> list[int]:
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    out = model(input_ids=ids, use_cache=True)
    past = out.past_key_values
    nxt = out.logits[:, -1, :].float().argmax(-1)
    toks = [int(nxt)]
    for _ in range(n_new - 1):
        out = model(input_ids=nxt.view(1, 1), past_key_values=past, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[:, -1, :].float().argmax(-1)
        toks.append(int(nxt))
    return toks


def divergence(ref: list[int], cand: list[int]) -> dict[str, Any]:
    n = min(len(ref), len(cand))
    first = next((i for i in range(n) if ref[i] != cand[i]), -1)
    mism = sum(1 for i in range(n) if ref[i] != cand[i])
    return {
        "n_compared": n,
        "first_divergence_idx": first,        # -1 == identical
        "num_mismatched": mism,
        "frac_mismatched": mism / n if n else 0.0,
        "identical": bool(mism == 0),
    }


# --------------------------------------------------------------------------- #
# Orchestration.
# --------------------------------------------------------------------------- #
def predicted_tps(b: float) -> float:
    syn = card.synthesize(group_size=card.SERVED_GROUP_SIZE)
    code4 = syn["served_decomp"]["code4_gb"]
    scale_meta = syn["served_decomp"]["scale_meta_gb"]
    non_body = syn["step_terms"]["non_body_gb"]
    if b >= 16:
        return float("nan")  # bf16 ref: not an int-N body read; TPS undefined here.
    return card.strict_nonspec_tps(b, code4, scale_meta, non_body, floor=True)


def run(args) -> dict[str, Any]:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[measure] WARNING: CUDA not available; running on CPU (slow). "
              "Set CUDA_VISIBLE_DEVICES=0.", flush=True)

    bits_list = [int(x) for x in str(args.bits).split(",") if x.strip()]
    records = read_ppl_records(Path(args.ppl_dataset))
    max_records = args.max_records if args.max_records > 0 else None

    print(f"[measure] base={args.base_model} scheme={args.scheme} g={args.group_size} "
          f"bits={bits_list} records={len(records)}(use {max_records or len(records)}) "
          f"greedy K={args.greedy_prompts} G={args.greedy_tokens}", flush=True)

    t0 = time.time()
    model, tok = load_model(args.base_model, device)
    print(f"[measure] model loaded in {time.time()-t0:.1f}s; "
          f"GPU mem {torch.cuda.memory_allocated()/2**30:.2f} GiB", flush=True)
    snap = snapshot_body(model)
    body_params = sum(t.numel() for t in snap.values())
    print(f"[measure] snapshotted {len(snap)} body linears, "
          f"{body_params/1e9:.3f}B params (kept lm_head+embed at source precision)", flush=True)

    greedy_prompts = [records[i]["ids"][: _prompt_len(records[i])]
                      for i in range(min(args.greedy_prompts, len(records)))]

    width_results: dict[int, dict[str, Any]] = {}
    greedy_tokens: dict[int, list[list[int]]] = {}
    for b in bits_list:
        tb = time.time()
        info = apply_width(model, snap, b, args.group_size, args.scheme, device)
        ppl = measure_ppl(model, records, device, max_records)
        gtoks = [greedy_decode(model, p, args.greedy_tokens, device) for p in greedy_prompts] \
            if args.greedy_prompts > 0 else []
        greedy_tokens[b] = gtoks
        width_results[b] = {
            "bits": b,
            "ppl": ppl["ppl"],
            "mean_record_ppl": ppl["mean_record_ppl"],
            "num_records": ppl["num_records"],
            "num_tokens": ppl["num_tokens"],
            "predicted_tps": predicted_tps(b),
            "ppl_pass": bool(ppl["ppl"] <= PPL_GATE),
            "body_info": info,
            "_per_record": ppl["per_record"],
        }
        print(f"[measure] b={b:>2}: PPL={ppl['ppl']:.4f} (mean_rec {ppl['mean_record_ppl']:.4f}) "
              f"pass<=2.42={width_results[b]['ppl_pass']} pred_tps={predicted_tps(b):.2f} "
              f"[{time.time()-tb:.1f}s]", flush=True)

    # greedy identity vs the int4 (b=4) baseline.
    ref_bits = 4 if 4 in greedy_tokens else bits_list[0]
    identity: dict[str, Any] = {"ref_bits": ref_bits, "per_width": {}}
    for b in bits_list:
        if b == ref_bits or not greedy_tokens.get(b):
            continue
        per_prompt = [divergence(r, c)
                      for r, c in zip(greedy_tokens[ref_bits], greedy_tokens[b])]
        all_ident = all(d["identical"] for d in per_prompt)
        tot_mis = sum(d["num_mismatched"] for d in per_prompt)
        tot_cmp = sum(d["n_compared"] for d in per_prompt)
        identity["per_width"][b] = {
            "identical_to_int4": bool(all_ident),
            "num_prompts": len(per_prompt),
            "total_mismatched": tot_mis,
            "total_compared": tot_cmp,
            "frac_mismatched": tot_mis / tot_cmp if tot_cmp else 0.0,
            "per_prompt": per_prompt,
        }
        width_results[b]["greedy_identical_to_int4"] = bool(all_ident)
        print(f"[measure] greedy identity b={b} vs int4: identical={all_ident} "
              f"mismatch={tot_mis}/{tot_cmp} ({100*tot_mis/max(tot_cmp,1):.2f}%)", flush=True)

    # --- PPL gate-transfer calibration ------------------------------------- #
    # The official gate runs on the organizer's PRIVATE set, where the deployed
    # int4 reads SERVED_INT4_PPL_SPEC=2.3772 and the cap is 2.42 -> only +1.81%
    # RELATIVE headroom. Our offline PPL is on the local 128-record proxy at a
    # different absolute scale (offline RTN-int4 ~ served-local 2.0188), so a raw
    # offline-abs <= 2.42 test is NOT the gate. The transferable quantity is the
    # RELATIVE increase over the offline int4 anchor; apply it to the deployed
    # anchor to predict the gate-set PPL. (Assumes the degradation RATIO transfers
    # across NL corpora -- the standard, best-available calibration without the
    # private set.)
    int4_ppl = width_results.get(4, {}).get("ppl")
    for b in bits_list:
        wr = width_results[b]
        if int4_ppl and b < 16:
            r = wr["ppl"] / int4_ppl
            wr["rel_increase_over_int4"] = r
            wr["gate_transfer_ppl_deployed"] = SERVED_INT4_PPL_SPEC * r
            wr["gate_transfer_pass"] = bool(SERVED_INT4_PPL_SPEC * r <= PPL_GATE)
        else:
            wr["rel_increase_over_int4"] = float("nan")
            wr["gate_transfer_ppl_deployed"] = float("nan")
            wr["gate_transfer_pass"] = bool(wr["ppl_pass"])

    # GO condition per width and overall.
    # BINDING gates: (i) PPL via the gate-transfer test (the real, dataset-faithful
    # quality gate) and (ii) the BW lift pred_tps>165.44. The official #319 greedy
    # gate is SELF-REFERENTIAL (served == own plain greedy AR), so a deterministic
    # sub-int4 checkpoint passes ITS OWN gate -- it is NOT required to match int4.
    # greedy_identical_to_int4 is therefore reported as a quality-DRIFT diagnostic
    # (even bf16 diverges ~84% from int4 on this near-tie-dense model), not a hard
    # gate; quality is already bounded by the PPL transfer.
    go_widths = []
    for b in bits_list:
        wr = width_results[b]
        is_subint4 = b < 4
        ppl_ok = wr["gate_transfer_pass"]
        tps_ok = bool((wr["predicted_tps"] > STRICT_NONSPEC_FLOOR_TPS)
                      if not math.isnan(wr["predicted_tps"]) else False)
        go = bool(is_subint4 and ppl_ok and tps_ok)
        wr["frontier_go"] = go
        if go:
            go_widths.append(b)
    any_go = bool(go_widths)

    return {
        "config": {
            "base_model": args.base_model, "scheme": args.scheme,
            "group_size": args.group_size, "bits": bits_list,
            "max_records": max_records or len(records),
            "greedy_prompts": args.greedy_prompts, "greedy_tokens": args.greedy_tokens,
            "ppl_gate": PPL_GATE, "strict_nonspec_floor_tps": STRICT_NONSPEC_FLOOR_TPS,
            "served_int4_ppl_nonspec_anchor": SERVED_INT4_PPL_NONSPEC,
            "served_int4_ppl_spec_anchor": SERVED_INT4_PPL_SPEC,
            "body_params_b": body_params / 1e9,
            "device": device,
        },
        "width_results": {str(b): width_results[b] for b in bits_list},
        "greedy_identity": identity,
        "frontier_go_widths": go_widths,
        "any_subint4_width_go": any_go,
    }


def _prompt_len(rec: dict[str, Any]) -> int:
    """Greedy prompt = the record's context (the part before scored targets)."""
    return rec["score_start"]


# --------------------------------------------------------------------------- #
# Reporting + W&B.
# --------------------------------------------------------------------------- #
def print_report(res: dict[str, Any]) -> None:
    cfg = res["config"]
    print("\n" + "=" * 100, flush=True)
    print("SUB-INT4 BODY MEASUREMENT (PR #355, NON-spec lane) — MEASURED PPL + greedy-identity "
          "vs PREDICTED TPS", flush=True)
    print("=" * 100, flush=True)
    print(f"  base={cfg['base_model']}  scheme={cfg['scheme']} g{cfg['group_size']}  "
          f"records={cfg['max_records']}  body={cfg['body_params_b']:.3f}B params", flush=True)
    print(f"  served-int4 calibration anchors: non-spec {cfg['served_int4_ppl_nonspec_anchor']} "
          f"/ spec {cfg['served_int4_ppl_spec_anchor']};  PPL gate {cfg['ppl_gate']}", flush=True)
    print("  PPL=offline (local proxy); gate_ppl=rel-increase transferred onto deployed "
          "int4 anchor 2.3772; gate=<=2.42", flush=True)
    print("-" * 100, flush=True)
    print(f"  {'bits':>5} {'PPL':>9} {'rel/int4':>9} {'gate_ppl':>9} {'gate<=2.42':>10} "
          f"{'pred_tps':>9} {'drift!=int4':>11} {'GO':>5}", flush=True)
    for b in cfg["bits"]:
        wr = res["width_results"][str(b)]
        pt = wr["predicted_tps"]
        pts = "  n/a  " if math.isnan(pt) else f"{pt:8.2f}"
        r = wr.get("rel_increase_over_int4", float("nan"))
        rstr = "  ref  " if b == 4 else ("  n/a  " if math.isnan(r) else f"{r:8.4f}")
        gp = wr.get("gate_transfer_ppl_deployed", float("nan"))
        gpstr = "  ref  " if b == 4 else ("  n/a  " if math.isnan(gp) else f"{gp:8.4f}")
        gpass = "ref" if b == 4 else str(wr.get("gate_transfer_pass", "—"))
        # drift: frac of greedy tokens that differ from int4 (1 - identical)
        gi = res["greedy_identity"]["per_width"].get(b) or res["greedy_identity"]["per_width"].get(str(b))
        drift = "ref" if b == 4 else (f"{gi['frac_mismatched']:.3f}" if gi else "—")
        print(f"  {b:>5} {wr['ppl']:>9.4f} {rstr:>9} {gpstr:>9} {gpass:>10} {pts:>9} "
              f"{drift:>11} {str(wr.get('frontier_go', False)):>5}", flush=True)
    print("-" * 100, flush=True)
    print(f"  >>> any_subint4_width_go (gate_transfer_PPL<=2.42 AND pred_tps>165.44) = "
          f"{res['any_subint4_width_go']}  widths={res['frontier_go_widths']}", flush=True)
    print("  NB: #319 greedy gate is SELF-referential (deterministic sub-int4 passes its OWN "
          "AR); drift!=int4 is a quality diagnostic, subsumed by PPL.", flush=True)
    print("=" * 100, flush=True)


def maybe_log_wandb(args, payload: dict[str, Any]) -> None:
    if not getattr(args, "wandb_name", None):
        return
    try:
        import wandb  # noqa: F401
        if str(REPO_ROOT) not in sys.path:
            sys.path.append(str(REPO_ROOT))
        from scripts.wandb_logging import (  # noqa: E402
            finish_wandb, init_wandb_run, log_json_artifact, log_summary,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[measure] wandb unavailable: {exc}", flush=True)
        return

    res = payload["result"]
    cfg = res["config"]
    run = init_wandb_run(
        job_type="validity-gate",
        agent="lawine",
        name=args.wandb_name,
        group=args.wandb_group,
        tags=["validity-gate", "sub-int4-body-355", "non-spec-frontier", "measured-ppl",
              "greedy-identity", "fake-quant-rtn", "pr-355"],
        config={**cfg, "wandb_group": args.wandb_group},
    )
    if run is None:
        print("[measure] wandb: no run (no WANDB_API_KEY/mode) — skipping", flush=True)
        return
    summary: dict[str, Any] = {
        "any_subint4_width_go": int(bool(res["any_subint4_width_go"])),
        "num_go_widths": len(res["frontier_go_widths"]),
    }
    for b in cfg["bits"]:
        wr = res["width_results"][str(b)]
        summary[f"ppl_b{b}"] = wr["ppl"]
        summary[f"ppl_pass_b{b}"] = int(bool(wr["ppl_pass"]))
        if not math.isnan(wr.get("rel_increase_over_int4", float("nan"))):
            summary[f"rel_increase_over_int4_b{b}"] = wr["rel_increase_over_int4"]
            summary[f"gate_transfer_ppl_b{b}"] = wr["gate_transfer_ppl_deployed"]
            summary[f"gate_transfer_pass_b{b}"] = int(bool(wr["gate_transfer_pass"]))
        if not math.isnan(wr["predicted_tps"]):
            summary[f"pred_tps_b{b}"] = wr["predicted_tps"]
        if "greedy_identical_to_int4" in wr:
            summary[f"greedy_identical_b{b}"] = int(bool(wr["greedy_identical_to_int4"]))
        if b < 4:
            summary[f"frontier_go_b{b}"] = int(bool(wr.get("frontier_go", False)))
    for b, d in res["greedy_identity"]["per_width"].items():
        summary[f"greedy_frac_mismatch_b{b}"] = d["frac_mismatched"]
    summary["peak_mem_mib"] = payload["peak_mem_mib"]

    # Overlay the analytic card (the PREDICTION) + its self-test as PRIMARY, so the
    # single run carries both the original 0-GPU deliverables and the measured pivot.
    try:
        syn = card.synthesize(group_size=card.SERVED_GROUP_SIZE)
        st = card.self_test(syn)
        summary["sub_int4_body_ceiling_self_test_passes"] = int(bool(
            st.get("sub_int4_body_ceiling_self_test_passes", False)))
        summary["strict_nonspec_tps_at_int2"] = syn.get("strict_nonspec_tps_at_int2")
        summary["b_to_clear_500"] = syn.get("b_to_clear_500")
        summary["b_min_ppl_codebook"] = syn.get("b_min_ppl")
        summary["b_min_ppl_marlin_scalar"] = syn.get("b_min_ppl_marlin_scalar")
        summary["sub_int4_alone_clears_500"] = int(bool(syn.get("sub_int4_alone_clears_500", False)))
        summary["spec_plus_subint4_clears_500"] = int(bool(syn.get("spec_plus_subint4_clears_500", False)))
    except Exception as exc:  # noqa: BLE001
        print(f"[measure] card synthesis overlay failed: {exc}", flush=True)
    summary = {k: v for k, v in summary.items()
               if not (isinstance(v, float) and not math.isfinite(v))}
    log_summary(run, summary, step=0)
    log_json_artifact(run, name="sub_int4_body_355_measured", artifact_type="validity",
                      data=payload)
    finish_wandb(run)
    print(f"[measure] wandb logged: {len(summary)} metrics", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-model", "--base_model", dest="base_model", default=DEFAULT_BASE_MODEL)
    ap.add_argument("--bits", default="16,4,3,2", help="comma list of body bit-widths")
    ap.add_argument("--scheme", choices=["asym", "sym"], default="asym")
    ap.add_argument("--group-size", "--group_size", dest="group_size", type=int, default=128)
    ap.add_argument("--ppl-dataset", "--ppl_dataset", dest="ppl_dataset",
                    default=str(DEFAULT_PPL_DATASET))
    ap.add_argument("--max-records", "--max_records", dest="max_records", type=int, default=0,
                    help="0 = all records")
    ap.add_argument("--greedy-prompts", "--greedy_prompts", dest="greedy_prompts",
                    type=int, default=8)
    ap.add_argument("--greedy-tokens", "--greedy_tokens", dest="greedy_tokens",
                    type=int, default=256)
    ap.add_argument("--out-dir", dest="out_dir", default=str(HERE))
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name", default=None)
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group",
                    default="sub-int4-body-355")
    args = ap.parse_args(argv)

    t0 = time.time()
    res = run(args)
    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    peak_mib = (torch.cuda.max_memory_allocated() / 2**20) if torch.cuda.is_available() else 0.0
    payload = {
        "created_at": created_at, "pr": 355, "agent": "lawine",
        "kind": "sub-int4-body-355-measured",
        "elapsed_s": round(time.time() - t0, 1),
        "peak_mem_mib": round(peak_mib, 1),
        "result": res,
    }
    print_report(res)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "measure_subint4_results.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=float)
    print(f"[measure] wrote {out_path}  (elapsed {payload['elapsed_s']}s, "
          f"peak {payload['peak_mem_mib']} MiB)", flush=True)

    maybe_log_wandb(args, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
