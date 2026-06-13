#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Scatter-elimination correctness proof for lmhead12k compute_logits (PR #41).

Proves the scatter-to-full-vocab + full-vocab argmax in
``vllm_lmhead12k/model.py:compute_logits`` is REDUNDANT for greedy token
selection and can be replaced by ``kept_ids[argmax(partial)]`` with
TOKEN-IDENTICAL output.

THE PROOF (universal, data-independent)
---------------------------------------
Let ``partial[m, j]`` be the pruned logit for kept row ``j`` (whose original
vocab id is ``kept_ids[j]``).

  Path A (current served path):
      full = -inf everywhere;  full[m, kept_ids[j]] = partial[m, j]
      token_A = argmax_v full[m, v]
  Path B (proposed):
      token_B = kept_ids[ argmax_j partial[m, j] ]

Every non-kept column of ``full`` is ``-inf`` and ``partial`` is finite
(softcap bounds it to (-30, 30)), so the full-vocab maximum is always attained
at a kept column; therefore ``argmax_v full[m, v]`` is some ``kept_ids[j]``.
Restricted to kept columns ``full`` carries the identical ``partial`` values, so
the max VALUE equals ``max_j partial[m, j]`` in both paths -- only the tie-break
INDEX can differ. ``torch.argmax`` returns the FIRST index attaining the max.
``kept_ids`` is STRICTLY ASCENDING (verified below), so the smallest original id
among tied maxima equals ``kept_ids[smallest kept-row index among tied maxima]``.
Hence ``token_A == token_B`` for EVERY input, ties included.  QED.

The proof rests on two structural premises; this script verifies BOTH directly,
then confirms with a large real-weight numerical sweep:

  (1) ``kept_ids`` strictly ascending, no dups, in [0, full_vocab) -- the
      tie-break premise.
  (2) ``torch.argmax`` first-occurrence tie-break on THIS gpu/dtype -- crafted
      exact ties on both the [.,KEPT] and [.,FULL_VOCAB] tensors.
  (3) real-weight sweep: the EXACT served bf16 ``lm_head`` sliced to ``kept_ids``
      applied to many hidden-state samples at the tree verify widths M, plus an
      ADVERSARIAL tie-injection sweep that forces the tie path real logits never
      hit. Reports ``scatter_equivalent_count`` / ``total_positions`` /
      ``equivalence_rate``.

LOCAL ONLY, single GPU, no HF Job, no submission, no leaderboard touch.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics

FULL_VOCAB = 262144
HIDDEN = 2560
SOFTCAP = 30.0
SWEEP_M = [1, 7, 17, 25, 45]  # tree verify widths (match PR #37 reporting M)
INT4_GLOB = (
    "~/.cache/huggingface/hub/"
    "models--google--gemma-4-E4B-it-qat-w4a16-ct/snapshots/*/model.safetensors"
)


def load_kept_ids(path: str) -> list[int]:
    return json.load(open(path))["kept_ids"]


def verify_kept_structure(kept: list[int]) -> dict:
    n = len(kept)
    strict = all(kept[i] < kept[i + 1] for i in range(n - 1))
    return {
        "kept_size": n,
        "strictly_ascending": strict,
        "has_duplicates": n != len(set(kept)),
        "min_id": min(kept),
        "max_id": max(kept),
        "in_range": min(kept) >= 0 and max(kept) < FULL_VOCAB,
        "premise_holds": strict and n == len(set(kept))
        and min(kept) >= 0 and max(kept) < FULL_VOCAB,
    }


def _softcap(x, cap):
    import torch

    if not cap:
        return x
    return torch.tanh(x / cap) * cap


def verify_argmax_tiebreak(kept_ids_t, device, cap) -> dict:
    """Construct partial rows with deliberate exact ties and confirm that
    BOTH paths agree AND torch.argmax returns the first (smallest) tied index,
    for bf16 and fp32, over several tie patterns."""
    import torch

    KEPT = int(kept_ids_t.shape[0])
    out = {}
    # tie patterns expressed as kept-row indices that share the max value
    patterns = [
        [0, 1],
        [3, 100, 5000, KEPT - 1],
        [7, 7777],
        [KEPT - 2, KEPT - 1],
        [1, 2, 3, 4, 5],
    ]
    for dt in (torch.bfloat16, torch.float32):
        rows = []
        all_match = True
        all_first = True
        for tie_js in patterns:
            partial = torch.full((1, KEPT), -5.0, dtype=dt, device=device)
            for j in tie_js:
                partial[0, j] = 7.0
            partial = _softcap(partial.float(), cap).to(dt)
            idxB = int(partial.argmax(dim=-1)[0].item())
            tokB = int(kept_ids_t[idxB].item())
            full = torch.full((1, FULL_VOCAB), float("-inf"), dtype=dt, device=device)
            full.scatter_(1, kept_ids_t.unsqueeze(0), partial)
            tokA = int(full.argmax(dim=-1)[0].item())
            match = tokA == tokB
            first = idxB == min(tie_js)
            all_match &= match
            all_first &= first
            rows.append({
                "tie_kept_rows": tie_js,
                "pathB_kept_row": idxB,
                "pathB_token": tokB,
                "pathA_token": tokA,
                "match": match,
                "argmax_returned_first_occurrence": first,
                "smallest_tied_orig_id": min(int(kept_ids_t[j]) for j in tie_js),
            })
        out[str(dt).replace("torch.", "")] = {
            "all_match": all_match,
            "all_first_occurrence": all_first,
            "patterns": rows,
        }
    return out


def load_real_lm_head(kept_ids_t, device):
    """Slice the EXACT served bf16 lm_head (kept rows) from the cached int4
    W4A16 checkpoint -- the int4 base keeps lm_head bf16 (compressed-tensors
    ignore list), so this is byte-for-byte the served pruned head weight."""
    import torch
    from safetensors import safe_open

    sf = glob.glob(os.path.expanduser(INT4_GLOB))
    if not sf:
        raise FileNotFoundError(f"int4 lm_head not found via {INT4_GLOB}")
    with safe_open(sf[0], framework="pt") as f:
        W = f.get_tensor("lm_head.weight")  # [FULL_VOCAB, HIDDEN] bf16
    W = W.to(device)
    W_12k = W.index_select(0, kept_ids_t).contiguous()
    del W
    torch.cuda.empty_cache()
    return W_12k


def _both_paths(partial, kept_ids_t):
    """partial: [M, KEPT] -> (token_A from scatter+full argmax, token_B remap)."""
    import torch

    M = partial.shape[0]
    idxB = partial.argmax(dim=-1)
    tokB = kept_ids_t[idxB]
    full = torch.full((M, FULL_VOCAB), float("-inf"),
                      dtype=partial.dtype, device=partial.device)
    full.scatter_(1, kept_ids_t.unsqueeze(0).expand(M, -1), partial)
    tokA = full.argmax(dim=-1)
    return tokA, tokB


def real_weight_sweep(W_12k, kept_ids_t, Ms, target_per_M, scales, device, cap,
                      logits_fp32, seed) -> dict:
    import torch

    g = torch.Generator(device="cpu").manual_seed(seed)
    KEPT, H = W_12k.shape
    per_M = {}
    total = equal = ties = 0
    logit_absmax = 0.0
    for M in Ms:
        m_total = m_equal = m_ties = 0
        trials = max(1, target_per_M // (len(scales) * M))
        for scale in scales:
            for _ in range(trials):
                hs = (torch.randn(M, H, generator=g, dtype=torch.float32) * scale).to(
                    device, dtype=torch.bfloat16)
                with torch.inference_mode():
                    partial = torch.nn.functional.linear(hs, W_12k)  # [M, KEPT]
                    if logits_fp32:
                        partial = partial.float()
                    partial = _softcap(partial, cap)
                    tokA, tokB = _both_paths(partial, kept_ids_t)
                    eq = int((tokA == tokB).sum().item())
                    rowmax = partial.max(dim=-1, keepdim=True).values
                    nties = int(((partial == rowmax).sum(dim=-1) > 1).sum().item())
                    logit_absmax = max(logit_absmax,
                                       float(partial.abs().max().item()))
                m_total += M
                m_equal += eq
                m_ties += nties
        per_M[M] = {"positions": m_total, "equal": m_equal, "tie_rows": m_ties,
                    "rate": m_equal / m_total}
        total += m_total
        equal += m_equal
        ties += m_ties
    return {
        "total_positions": total,
        "scatter_equivalent_count": equal,
        "equivalence_rate": equal / total,
        "natural_tie_rows": ties,
        "logit_abs_max_observed": logit_absmax,
        "per_M": per_M,
        "scales": scales,
        "logits_fp32": logits_fp32,
    }


def adversarial_tie_sweep(W_12k, kept_ids_t, M, trials, device, cap, logits_fp32,
                          seed) -> dict:
    """Force the tie path real logits never reach: for each row copy the row-max
    value into a random OTHER kept column -> exact tie, then verify Path A ==
    Path B AND that the resolved token is the SMALLER original id of the tie."""
    import torch

    g = torch.Generator(device="cpu").manual_seed(seed)
    KEPT, H = W_12k.shape
    total = equal = correct_min = 0
    for _ in range(trials):
        hs = torch.randn(M, H, generator=g, dtype=torch.float32).to(
            device, dtype=torch.bfloat16)
        with torch.inference_mode():
            partial = torch.nn.functional.linear(hs, W_12k)
            if logits_fp32:
                partial = partial.float()
            partial = _softcap(partial, cap).clone()
            orig_arg = partial.argmax(dim=-1)  # [M]
            rowmax = partial.gather(1, orig_arg.unsqueeze(1)).squeeze(1)  # [M]
            inj = torch.randint(0, KEPT, (M,), generator=g).to(device)
            # ensure inj != orig_arg
            clash = inj == orig_arg
            inj[clash] = (inj[clash] + 1) % KEPT
            partial.scatter_(1, inj.unsqueeze(1), rowmax.unsqueeze(1).to(partial.dtype))
            tokA, tokB = _both_paths(partial, kept_ids_t)
            # expected: smaller original id among {orig_arg, inj}
            ida = kept_ids_t[orig_arg]
            idb = kept_ids_t[inj]
            expected = torch.minimum(ida, idb)
            equal += int((tokA == tokB).sum().item())
            correct_min += int((tokB == expected).sum().item())
            total += M
    return {
        "positions": total,
        "pathA_eq_pathB": equal,
        "pathB_eq_smaller_id": correct_min,
        "equivalence_rate": equal / total,
        "min_id_rate": correct_min / total,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kept-ids",
                    default="submissions/lmhead12k_empirical/kept_ids.json")
    ap.add_argument("--softcap", type=float, default=SOFTCAP)
    ap.add_argument("--logits-fp32", action="store_true", default=True,
                    help="vLLM upcasts logits to fp32 before softcap/argmax")
    ap.add_argument("--bf16-logits", dest="logits_fp32", action="store_false")
    ap.add_argument("--target-per-M", type=int, default=50000,
                    help="approx positions tested per M in the real-weight sweep")
    ap.add_argument("--scales", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    ap.add_argument("--adv-M", type=int, default=64)
    ap.add_argument("--adv-trials", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output",
                    default="research/spec_cost_model/lmhead12k_scatter_equiv.json")
    ap.add_argument("--wandb_project", "--wandb-project", dest="wandb_project",
                    default=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"))
    ap.add_argument("--wandb_entity", "--wandb-entity", dest="wandb_entity",
                    default=os.environ.get("WANDB_ENTITY"))
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group",
                    default="spec-verify-scatter-free")
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name", default=None)
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    import torch

    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    cap = args.softcap if args.softcap else None

    kept = load_kept_ids(args.kept_ids)
    struct = verify_kept_structure(kept)
    print("[scatter-equiv] ===== Premise (1): kept_ids structure =====", flush=True)
    print(json.dumps(struct, indent=2), flush=True)
    if not struct["premise_holds"]:
        raise SystemExit("[scatter-equiv] FATAL: kept_ids structure premise FAILED; "
                         "the universal proof does not apply -- STOP.")

    kept_ids_t = torch.tensor(kept, dtype=torch.long, device=device)

    print("\n[scatter-equiv] ===== Premise (2): torch.argmax tie-break =====",
          flush=True)
    tb = verify_argmax_tiebreak(kept_ids_t, device, cap)
    print(json.dumps(tb, indent=2), flush=True)

    print("\n[scatter-equiv] ===== Step (3a): real-weight equivalence sweep =====",
          flush=True)
    W_12k = load_real_lm_head(kept_ids_t, device)
    print(f"[scatter-equiv] served lm_head sliced: {tuple(W_12k.shape)} "
          f"{W_12k.dtype}", flush=True)
    sweep = real_weight_sweep(W_12k, kept_ids_t, SWEEP_M, args.target_per_M,
                              args.scales, device, cap, args.logits_fp32, args.seed)
    print(f"{'M':>3} {'positions':>10} {'equal':>10} {'tie_rows':>9} {'rate':>10}",
          flush=True)
    for M in SWEEP_M:
        r = sweep["per_M"][M]
        print(f"{M:>3} {r['positions']:>10} {r['equal']:>10} {r['tie_rows']:>9} "
              f"{r['rate']:>10.6f}", flush=True)
    print(f"[scatter-equiv] TOTAL: equivalent {sweep['scatter_equivalent_count']}"
          f"/{sweep['total_positions']} = {sweep['equivalence_rate']:.8f}  "
          f"(natural ties {sweep['natural_tie_rows']}, "
          f"logit|max| {sweep['logit_abs_max_observed']:.3f})", flush=True)

    print("\n[scatter-equiv] ===== Step (3b): adversarial tie injection =====",
          flush=True)
    adv = adversarial_tie_sweep(W_12k, kept_ids_t, args.adv_M, args.adv_trials,
                                device, cap, args.logits_fp32, args.seed + 1)
    print(json.dumps(adv, indent=2), flush=True)

    verdict = (
        struct["premise_holds"]
        and all(d["all_match"] and d["all_first_occurrence"] for d in tb.values())
        and sweep["equivalence_rate"] == 1.0
        and adv["equivalence_rate"] == 1.0
        and adv["min_id_rate"] == 1.0
    )
    payload = {
        "verdict": "EQUIVALENT" if verdict else "DIVERGENT",
        "kept_structure": struct,
        "argmax_tiebreak": tb,
        "real_weight_sweep": sweep,
        "adversarial_tie_sweep": adv,
        "config": {
            "softcap": args.softcap, "logits_fp32": args.logits_fp32,
            "target_per_M": args.target_per_M, "scales": args.scales,
            "sweep_M": SWEEP_M, "adv_M": args.adv_M, "adv_trials": args.adv_trials,
            "kept_ids": args.kept_ids, "full_vocab": FULL_VOCAB, "hidden": HIDDEN,
        },
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    json.dump(payload, open(args.output, "w"), indent=2)
    print(f"\n[scatter-equiv] VERDICT: {payload['verdict']}", flush=True)
    print(f"[scatter-equiv] wrote {args.output}", flush=True)

    if not args.no_wandb and args.wandb_name:
        try:
            _log_wandb(args, payload)
        except Exception as e:  # noqa: BLE001
            print(f"[scatter-equiv] W&B logging failed (non-fatal): {e!r}", flush=True)

    del W_12k
    torch.cuda.empty_cache()
    print("[scatter-equiv] DONE", flush=True)
    raise SystemExit(0 if verdict else 1)


def _log_wandb(args, payload):
    import wandb

    run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                     group=args.wandb_group, name=args.wandb_name,
                     job_type="correctness-proof", config=payload["config"])
    sweep = payload["real_weight_sweep"]
    adv = payload["adversarial_tie_sweep"]
    summary = {
        "verdict_equivalent": payload["verdict"] == "EQUIVALENT",
        "kept_strictly_ascending": payload["kept_structure"]["strictly_ascending"],
        "scatter_equivalent_count": sweep["scatter_equivalent_count"],
        "total_positions": sweep["total_positions"],
        "scatter_equivalence_rate": sweep["equivalence_rate"],
        "natural_tie_rows": sweep["natural_tie_rows"],
        "logit_abs_max_observed": sweep["logit_abs_max_observed"],
        "adv_tie_equivalence_rate": adv["equivalence_rate"],
        "adv_tie_min_id_rate": adv["min_id_rate"],
        "adv_tie_positions": adv["positions"],
    }
    for M in SWEEP_M:
        summary[f"equiv_rate_M{M}"] = sweep["per_M"][M]["rate"]
    cols = ["M", "positions", "equal", "tie_rows", "rate"]
    tbl = wandb.Table(columns=cols)
    for M in SWEEP_M:
        r = sweep["per_M"][M]
        tbl.add_data(M, r["positions"], r["equal"], r["tie_rows"], r["rate"])
    run.log({"equivalence_by_M": tbl})
    run.summary.update(summary)
    run.finish()
    print(f"[scatter-equiv] W&B run: {run.url}", flush=True)


if __name__ == "__main__":
    main()
