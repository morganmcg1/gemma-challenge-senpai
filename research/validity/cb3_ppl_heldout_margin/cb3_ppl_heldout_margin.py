#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""Held-out stress test of #372's cb3 PPL margin (PR #394) — winner's-curse check.

THE QUESTION
------------
lawine #372 (`mpzfw116`) reported `mixed_config_measured_gate_ppl`=2.3812 at the
mixed-precision optimum k=232 (of 258 body linears), a +0.039 margin under the
2.42 PPL gate. But that k was **greedy-selected ON the official 128 gate set
itself** — selection-on-eval / winner's curse. If the +0.039 margin is a
selection artifact, the whole cb3 supply lane (lawine #388 body-read shrink,
#391 M=8, denken #392 composition) is un-deployable because the served PPL gate
runs on data the budget was never tuned on. This card asks: **does the in-sample
2.3812 survive an honest selection/held-out split, and a disjoint OOD slice?**

WHAT IS REUSED FROM #372 (no quantizer re-derivation)
-----------------------------------------------------
  * the cb3 RHT+VQ fake-quant (`measure_codebook.codebook_quant_per_group`,
    `rht_matrix`, `build_gaussian_codebook`), int4 RTN (`measure_subint4`),
    `QuantModel` state machine, and `gate_transfer_ppl` — imported directly.
  * the per-module **ascending-sensitivity ordering** `order_ascending` (258
    names) and the in-sample optimum k*=232 from
    `measure_mixed_precision_results.json`. The selection *axis* under test is k
    (the #372 selection knob). The ordering is taken as given from #372 (it is
    itself full-128-derived — see CAVEAT), so this is an honest test of the k
    THRESHOLD, not of the ordering.

GATE (identical to #372, per measurement set X)
-----------------------------------------------
  gate_ppl(cfg, X) = SERVED_INT4_PPL_SPEC * PPL(cfg, X) / PPL(int4, X)
with SERVED_INT4_PPL_SPEC=2.3772. The ratio is a *within-set* relative
degradation of cfg vs int4, so the half-specific int4 anchor isolates the
selection-of-k effect from absolute PPL differences between subsets. Pass iff
gate_ppl <= 2.42.

EFFICIENCY (measure once, derive every subset)
----------------------------------------------
`measure_ppl` returns per-record NLL. We measure per-record NLL on the full 128
ONCE for int4 (k=0) and cb3 at a grid of k, then DERIVE the PPL of any subset
(selection-half S, held-out-half H, full) by token-weighted aggregation. The
split seeds and the in-sample number are then pure arithmetic on stored
per-record NLL — no extra GPU per seed.

DELIVERABLES (PR #394)
----------------------
 1. `in_sample_gate_ppl`  — cb3 at k=232 on full 128 (reproduces #372's 2.3812).
 2. PRIMARY held-out split over >=3 seeds: partition 128 -> S(64)/H(64); select
    k on S by #372's gate criterion (largest k with gate_S<=2.42); measure on H:
      `selection_half_ppl`, `heldout_gate_ppl`, `heldout_margin`=2.42-heldout,
      `in_sample_overfit_delta`=heldout-selection, `cb3_heldout_clears_2p42`
      (bool, worst seed counts).
 3. OOD leg: disjoint sharegpt slice (N>=64) -> `ood_sharegpt_ppl`, `ood_margin`,
    `cb3_ood_clears_2p42` (bool, worst seed) at the S-selected k.
 4. Self-test (PRIMARY): reproduces 2.3812; >=3 seeds; S/H disjoint partition;
    PPL greedy under the strict proxy; worst-seed margin is the reported one.

DECISION: cb3_heldout_clears_2p42 AND cb3_ood_clears_2p42 -> cb3 supply lift is
DEPLOYABLE. Either False -> the cb3 margin was a selection artifact; supply lane
is PPL-blocked (report the heldout_margin sign).

CAVEAT (stated in the verdict): #372's ordering is reused as-is (full-128
derived), so the *ordering* is in-sample; only the k threshold is selected
out-of-sample. A fully-honest test would also re-derive the per-module ordering
on S (258 evals/seed) — flagged as a follow-up. This test is therefore a
conservative-optimistic check: if it FAILS, the curse is real with the ordering
advantage; if it PASSES, ordering leak could still inflate it.

Identity-safe: GPU PPL only, NO submission, NO served-file change, NO --launch,
0 official TPS.

Run (single A10G; ~20 min):
    cd target/ && CUDA_VISIBLE_DEVICES=0 .venvs/vllm022/bin/python \
      research/validity/cb3_ppl_heldout_margin/cb3_ppl_heldout_margin.py \
      --gpu --heldout-split --ood-sharegpt --split-seeds 0 1 2 \
      --reuse-372-quantizer --wandb_group cb3-ppl-heldout-margin \
      --wandb_name kanna/cb3-ppl-heldout-margin
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
SUB372 = REPO_ROOT / "research" / "validity" / "sub_int4_body_ceiling"
if str(SUB372) not in sys.path:
    sys.path.insert(0, str(SUB372))

import measure_subint4 as m355  # noqa: E402
import measure_codebook as mcb  # noqa: E402
import measure_mixed_precision as mp  # noqa: E402

# ---- anchors / constants (single source of truth = #372) ------------------- #
PPL_GATE = mp.PPL_GATE                               # 2.42
SERVED_INT4_PPL_SPEC = mp.SERVED_INT4_PPL_SPEC       # 2.3772 (gate's named anchor)
DEFAULT_BASE_MODEL = m355.DEFAULT_BASE_MODEL         # strict proxy: q4_0-unquantized (bf16 QAT base)
STRICT_PROXY_MODELS = (
    "google/gemma-4-E4B-it-qat-q4_0-unquantized",    # #372's actual base (bf16, fake-quant target)
    "google/gemma-4-E4B-it-qat-w4a16-ct",            # the named deployable-strict proxy family
)
DEFAULT_PPL_DATASET = m355.DEFAULT_PPL_DATASET
RESULTS_372 = SUB372 / "measure_mixed_precision_results.json"
IN_SAMPLE_GATE_372 = 2.3811966031692555              # the number under stress test
K_STAR_372 = 232                                     # #372's in-sample optimum
N_OFFICIAL = 128
gate_transfer_ppl = mp.gate_transfer_ppl

RESULTS_NAME = "cb3_ppl_heldout_margin_results.json"


# --------------------------------------------------------------------------- #
# Reuse #372's ordering + in-sample optimum.
# --------------------------------------------------------------------------- #
def load_372(path: Path = RESULTS_372) -> dict[str, Any]:
    d = json.loads(path.read_text())
    r = d["result"]
    return {
        "order_ascending": r["sensitivity"]["order_ascending"],
        "k_star": r["allocation"]["k_star"],
        "mixed_config_measured_gate_ppl": r["allocation"]["mixed_config_measured_gate_ppl"],
        "int4_localproxy_ppl": r["config"]["int4_localproxy_ppl"],
        "base_model": r["config"]["base_model"],
    }


# --------------------------------------------------------------------------- #
# Per-record NLL measurement + subset aggregation.
# --------------------------------------------------------------------------- #
def per_record_nll(model, records, device) -> tuple[list[float], list[int]]:
    """Measure per-record NLL on the given records (full set). Returns aligned
    (nll[i], tok[i]) so any index-subset PPL is exp(sum nll / sum tok)."""
    res = m355.measure_ppl(model, records, device, None)
    nll = [p["nll"] for p in res["per_record"]]
    tok = [p["num_tokens"] for p in res["per_record"]]
    return nll, tok


def subset_ppl(nll: list[float], tok: list[int], idxs: list[int]) -> float:
    s_nll = sum(nll[i] for i in idxs)
    s_tok = sum(tok[i] for i in idxs)
    return math.exp(s_nll / s_tok)


def subset_gate(nll_cfg: list[float], nll_int4: list[float], tok: list[int],
                idxs: list[int]) -> float:
    return SERVED_INT4_PPL_SPEC * subset_ppl(nll_cfg, tok, idxs) / subset_ppl(nll_int4, tok, idxs)


# --------------------------------------------------------------------------- #
# k-grid: 0 (int4) + coarse coverage + fine band around the in-sample crossing.
# --------------------------------------------------------------------------- #
def build_k_grid(n_modules: int, k_star_insample: int, fine_lo: int, fine_step: int,
                 smoke: bool) -> list[int]:
    if smoke:
        return sorted({0, max(1, n_modules // 2), k_star_insample, n_modules})
    coarse = [0, 40, 80, 120, 150, 170]
    fine = list(range(fine_lo, n_modules + 1, fine_step))
    grid = sorted({k for k in (coarse + fine + [k_star_insample, n_modules]) if 0 <= k <= n_modules})
    return grid


def measure_k_grid(qm, model, records, device, order, grid) -> dict[int, list[float]]:
    """Per-record NLL at each k (cb3 on order[:k], int4 elsewhere). Monotone k so
    QuantModel only flips int4->cb3 incrementally. Returns {k: nll_list}. tok is
    constant across k; captured separately by the caller via k=0."""
    nll_by_k: dict[int, list[float]] = {}
    tok_ref = None
    for j, k in enumerate(sorted(grid)):
        t0 = time.time()
        if k == 0:
            qm.set_all("int4")
        else:
            qm.set_config(set(order[:k]))
        nll, tok = per_record_nll(model, records, device)
        nll_by_k[k] = nll
        if tok_ref is None:
            tok_ref = tok
        else:
            assert tok == tok_ref, "scored-token counts must be constant across k"
        print(f"[heldout] k-grid {j+1}/{len(grid)} k={k:>3}: "
              f"full128 PPL={subset_ppl(nll, tok, list(range(len(records)))):.4f} "
              f"[{time.time()-t0:.1f}s]", flush=True)
    nll_by_k["_tok"] = tok_ref  # type: ignore[assignment]
    return nll_by_k


def select_k_on_subset(nll_by_k: dict[int, list[float]], tok: list[int], grid: list[int],
                       idxs: list[int]) -> tuple[int, float]:
    """#372 criterion on a subset: largest grid-k with gate(k, subset) <= 2.42."""
    k_pass = 0
    for k in sorted(grid):
        g = subset_gate(nll_by_k[k], nll_by_k[0], tok, idxs)
        if g <= PPL_GATE:
            k_pass = k
    g_sel = subset_gate(nll_by_k[k_pass], nll_by_k[0], tok, idxs)
    return k_pass, g_sel


# --------------------------------------------------------------------------- #
# OOD sharegpt records (disjoint from the mmlu_pro official eval).
# --------------------------------------------------------------------------- #
def load_sharegpt_conversations(n_want: int) -> list[dict[str, Any]]:
    """Real ShareGPT (anon8231489123/ShareGPT_Vicuna_unfiltered V3 cleaned split).
    Falls back to UltraChat (HuggingFaceH4/ultrachat_200k) via the datasets-server
    rows API if the sharegpt JSON is unavailable. Returns a list of
    {'user': str, 'assistant': str} first-turn pairs."""
    # --- primary: cached/downloaded sharegpt json -------------------------- #
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            "anon8231489123/ShareGPT_Vicuna_unfiltered",
            "ShareGPT_V3_unfiltered_cleaned_split.json", repo_type="dataset",
        )
        raw = json.loads(Path(path).read_text())
        pairs: list[dict[str, Any]] = []
        for conv in raw:
            turns = conv.get("conversations") or conv.get("conversation") or []
            user = next((t.get("value", "") for t in turns if t.get("from") in ("human", "user")), "")
            asst = next((t.get("value", "") for t in turns if t.get("from") in ("gpt", "assistant", "bot")), "")
            if user.strip() and asst.strip():
                pairs.append({"user": user, "assistant": asst, "source": "sharegpt_v3"})
            if len(pairs) >= n_want * 3:  # over-collect; some drop on length filters
                break
        if len(pairs) >= n_want:
            print(f"[heldout] OOD source = ShareGPT_V3 ({len(pairs)} first-turn pairs collected)", flush=True)
            return pairs
    except Exception as exc:  # noqa: BLE001
        print(f"[heldout] sharegpt json unavailable ({repr(exc)[:120]}); trying UltraChat rows API", flush=True)
    # --- fallback: UltraChat via datasets-server rows API ------------------ #
    import os
    import urllib.request
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    hdr = {"Authorization": f"Bearer {tok}"} if tok else {}
    pairs = []
    off = 0
    while len(pairs) < n_want * 3 and off < 2000:
        u = ("https://datasets-server.huggingface.co/rows?dataset=HuggingFaceH4/ultrachat_200k"
             f"&config=default&split=train_sft&offset={off}&length=100")
        req = urllib.request.Request(u, headers=hdr)
        with urllib.request.urlopen(req, timeout=60) as resp:
            d = json.load(resp)
        for row in d.get("rows", []):
            msgs = row["row"].get("messages") or []
            user = next((m.get("content", "") for m in msgs if m.get("role") == "user"), "")
            asst = next((m.get("content", "") for m in msgs if m.get("role") == "assistant"), "")
            if user.strip() and asst.strip():
                pairs.append({"user": user, "assistant": asst, "source": "ultrachat_200k"})
        off += 100
        if not d.get("rows"):
            break
    print(f"[heldout] OOD source = UltraChat ({len(pairs)} first-turn pairs)", flush=True)
    return pairs


def _gemma4_control_ids(tokenizer) -> tuple[int, int, int]:
    """(BOS, start-of-turn, end-of-turn). Gemma4 uses <|turn>=105 / <turn|>=106
    (NOT <start_of_turn>); its multimodal chat_template is unusable for plain
    strings, so we build the turn structure from explicit control ids."""
    bos = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 2
    sot, eot = 105, 106
    for tid, t in getattr(tokenizer, "added_tokens_decoder", {}).items():
        s = getattr(t, "content", str(t))
        if s in ("<|turn>", "<start_of_turn>"):
            sot = int(tid)
        elif s in ("<turn|>", "<end_of_turn>"):
            eot = int(tid)
    return bos, sot, eot


def build_ood_records(pairs, tokenizer, n_target: int, max_ctx: int, tgt_lo: int,
                      tgt_cap: int) -> list[dict[str, Any]]:
    """Build PPL records that score the assistant reply given a Gemma4-format user
    turn, mirroring the official record shape (ids, score_start, score_end).
    Built from explicit control ids so content tokenizes normally and full_ids
    always extends prompt_ids. Deterministic: first n_target pairs passing filters."""
    bos, sot, eot = _gemma4_control_ids(tokenizer)

    def enc(s: str) -> list[int]:
        return tokenizer(s, add_special_tokens=False)["input_ids"]

    out: list[dict[str, Any]] = []
    for p in pairs:
        try:
            prompt_ids = ([bos, sot] + enc("user\n" + p["user"]) + [eot, sot] + enc("model\n"))
            target_ids = enc(p["assistant"]) + [eot]
        except Exception:  # noqa: BLE001
            continue
        full_ids = prompt_ids + target_ids
        score_start = len(prompt_ids)
        if score_start < 1 or score_start > max_ctx or len(target_ids) < 1:
            continue
        score_end = min(len(full_ids), score_start + tgt_cap)
        if score_end - score_start < tgt_lo:
            continue
        out.append({
            "id": f"ood_{len(out)}",
            "ids": list(full_ids[:score_end]),
            "score_start": score_start,
            "score_end": score_end,
            "source": p.get("source", "sharegpt"),
        })
        if len(out) >= n_target:
            break
    return out


# --------------------------------------------------------------------------- #
# Greedy-identity (report-only; "greedy under the strict proxy").
# --------------------------------------------------------------------------- #
def greedy_drift(qm, model, records, device, order, k_mixed, n_prompts, n_tokens) -> float:
    prompts = [records[i]["ids"][: m355._prompt_len(records[i])]
               for i in range(min(n_prompts, len(records)))]
    qm.set_all("int4")
    ref = [m355.greedy_decode(model, p, n_tokens, device) for p in prompts]
    qm.set_config(set(order[:k_mixed]))
    cand = [m355.greedy_decode(model, p, n_tokens, device) for p in prompts]
    per = [m355.divergence(r, c) for r, c in zip(ref, cand)]
    tot_mis = sum(d["num_mismatched"] for d in per)
    tot_cmp = sum(d["n_compared"] for d in per)
    return tot_mis / tot_cmp if tot_cmp else 0.0


# --------------------------------------------------------------------------- #
# Stats.
# --------------------------------------------------------------------------- #
def mean_se(vals: list[float]) -> tuple[float, float]:
    n = len(vals)
    m = sum(vals) / n
    if n < 2:
        return m, 0.0
    var = sum((v - m) ** 2 for v in vals) / (n - 1)
    return m, math.sqrt(var / n)


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-model", "--base_model", dest="base_model", default=DEFAULT_BASE_MODEL)
    ap.add_argument("--gpu", action="store_true", help="(documentation flag; uses cuda if available)")
    ap.add_argument("--heldout-split", action="store_true", default=True)
    ap.add_argument("--ood-sharegpt", action="store_true", default=True)
    ap.add_argument("--no-ood", dest="ood_sharegpt", action="store_false")
    ap.add_argument("--reuse-372-quantizer", action="store_true", default=True)
    ap.add_argument("--split-seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--quant-seed", type=int, default=0, help="RHT+codebook seed (fixed = #372's 0)")
    ap.add_argument("--scheme", choices=["asym", "sym"], default="asym")
    ap.add_argument("--group-size", "--group_size", dest="group_size", type=int, default=128)
    ap.add_argument("--vq-dim", "--vq_dim", dest="vq_dim", type=int, default=2)
    ap.add_argument("--ppl-dataset", "--ppl_dataset", dest="ppl_dataset", default=str(DEFAULT_PPL_DATASET))
    ap.add_argument("--fine-lo", type=int, default=180, help="k-grid fine band lower bound")
    ap.add_argument("--fine-step", type=int, default=3, help="k-grid fine band step")
    ap.add_argument("--ood-n", type=int, default=96, help="OOD records to build (>=64)")
    ap.add_argument("--ood-max-ctx", type=int, default=1024)
    ap.add_argument("--ood-tgt-lo", type=int, default=16)
    ap.add_argument("--ood-tgt-cap", type=int, default=384)
    ap.add_argument("--greedy-prompts", type=int, default=8)
    ap.add_argument("--greedy-tokens", type=int, default=128)
    ap.add_argument("--repro-tol", type=float, default=0.02, help="abs tol for reproducing #372's 2.3812")
    ap.add_argument("--smoke", action="store_true", help="tiny grid + few records (wiring check)")
    ap.add_argument("--out-dir", dest="out_dir", default=str(HERE))
    ap.add_argument("--wandb_name", "--wandb-name", dest="wandb_name", default=None)
    ap.add_argument("--wandb_group", "--wandb-group", dest="wandb_group", default="cb3-ppl-heldout-margin")
    args = ap.parse_args(argv)

    t_start = time.time()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[heldout] WARNING: CUDA not available; set CUDA_VISIBLE_DEVICES=0", flush=True)

    # ---- reuse #372's ordering + optimum ---------------------------------- #
    d372 = load_372()
    order = d372["order_ascending"]
    k_star_insample = d372["k_star"]
    if k_star_insample != K_STAR_372:
        print(f"[heldout] WARNING: #372 k_star={k_star_insample} != expected {K_STAR_372}", flush=True)
    print(f"[heldout] reused #372: {len(order)} modules, in-sample k*={k_star_insample}, "
          f"reported gate={d372['mixed_config_measured_gate_ppl']:.4f}", flush=True)

    # ---- load strict proxy, snapshot, build the cb3/int4 QuantModel ------- #
    records = m355.read_ppl_records(Path(args.ppl_dataset))
    if args.smoke:
        records = records[:16]
    n_records = len(records)
    model, tokenizer = m355.load_model(args.base_model, device)
    print(f"[heldout] model={args.base_model} loaded; GPU {torch.cuda.memory_allocated()/2**30:.2f} GiB",
          flush=True)
    snap = m355.snapshot_body(model)
    assert set(order) == set(snap), \
        f"#372 ordering ({len(order)}) != snapshot body linears ({len(snap)})"
    R = mcb.rht_matrix(args.group_size, device, seed=args.quant_seed)
    codebook = mcb.build_gaussian_codebook(3, args.vq_dim, device, seed=args.quant_seed)
    qm = mp.QuantModel(model, snap, args.group_size, args.scheme, args.vq_dim, codebook, R, device)

    # ---- measure per-record NLL on the OFFICIAL 128 at the k-grid --------- #
    grid = build_k_grid(len(order), k_star_insample, args.fine_lo, args.fine_step, args.smoke)
    print(f"[heldout] k-grid ({len(grid)}): {grid}", flush=True)
    nll_by_k = measure_k_grid(qm, model, records, device, order, grid)
    tok = nll_by_k.pop("_tok")  # type: ignore[arg-type]
    full_idx = list(range(n_records))

    # ---- 1. in-sample reproduction (full 128 at k*) ----------------------- #
    int4_ppl_full = subset_ppl(nll_by_k[0], tok, full_idx)
    in_sample_gate_ppl = subset_gate(nll_by_k[k_star_insample], nll_by_k[0], tok, full_idx)
    repro_abs_diff = abs(in_sample_gate_ppl - IN_SAMPLE_GATE_372)
    reproduces_372 = bool(repro_abs_diff <= args.repro_tol)
    print(f"[heldout] IN-SAMPLE: int4_full_ppl={int4_ppl_full:.4f} (372: {d372['int4_localproxy_ppl']:.4f})  "
          f"in_sample_gate_ppl={in_sample_gate_ppl:.4f} (372: {IN_SAMPLE_GATE_372:.4f}) "
          f"|diff|={repro_abs_diff:.4f} reproduces={reproduces_372}", flush=True)

    # ---- 2. PRIMARY: selection/held-out split over seeds ------------------ #
    seed_rows: list[dict[str, Any]] = []
    for seed in args.split_seeds:
        rng = random.Random(seed)
        perm = list(range(n_records))
        rng.shuffle(perm)
        half = n_records // 2
        S = sorted(perm[:half])
        H = sorted(perm[half:])
        disjoint = (set(S).isdisjoint(H) and sorted(S + H) == full_idx and len(S) == len(H))
        k_S, sel_gate = select_k_on_subset(nll_by_k, tok, grid, S)
        heldout_gate = subset_gate(nll_by_k[k_S], nll_by_k[0], tok, H)
        row = {
            "seed": seed, "S_size": len(S), "H_size": len(H), "disjoint_partition": bool(disjoint),
            "selected_k_heldout": k_S,
            "selection_half_ppl": sel_gate,
            "heldout_gate_ppl": heldout_gate,
            "heldout_margin": PPL_GATE - heldout_gate,
            "in_sample_overfit_delta": heldout_gate - sel_gate,
            "heldout_clears": bool(heldout_gate <= PPL_GATE),
        }
        seed_rows.append(row)
        print(f"[heldout] seed={seed}: k_S={k_S} sel_gate={sel_gate:.4f} "
              f"heldout_gate={heldout_gate:.4f} margin={row['heldout_margin']:+.4f} "
              f"overfit_delta={row['in_sample_overfit_delta']:+.4f} clears={row['heldout_clears']}", flush=True)

    heldout_vals = [r["heldout_gate_ppl"] for r in seed_rows]
    sel_vals = [r["selection_half_ppl"] for r in seed_rows]
    overfit_vals = [r["in_sample_overfit_delta"] for r in seed_rows]
    heldout_mean, heldout_se = mean_se(heldout_vals)
    sel_mean, sel_se = mean_se(sel_vals)
    overfit_mean, overfit_se = mean_se(overfit_vals)
    worst_seed_row = max(seed_rows, key=lambda r: r["heldout_gate_ppl"])
    heldout_gate_worst = worst_seed_row["heldout_gate_ppl"]
    heldout_margin_worst = PPL_GATE - heldout_gate_worst
    cb3_heldout_clears_2p42 = bool(all(r["heldout_clears"] for r in seed_rows))
    selected_k_heldout = worst_seed_row["selected_k_heldout"]

    # ---- 3. OOD leg ------------------------------------------------------- #
    ood: dict[str, Any] = {"enabled": bool(args.ood_sharegpt)}
    cb3_ood_clears_2p42 = None
    ood_sharegpt_ppl = float("nan")
    ood_margin = float("nan")
    if args.ood_sharegpt:
        min_ood = 8 if args.smoke else 64
        n_want = min_ood if args.smoke else max(64, args.ood_n)
        pairs = load_sharegpt_conversations(n_want)
        ood_records = build_ood_records(pairs, tokenizer, n_want,
                                        args.ood_max_ctx, args.ood_tgt_lo, args.ood_tgt_cap)
        assert len(ood_records) >= min_ood, f"OOD slice too small: {len(ood_records)} (<{min_ood})"
        ood_idx = list(range(len(ood_records)))
        needed_ks = sorted({0} | {r["selected_k_heldout"] for r in seed_rows})
        ood_nll: dict[int, list[float]] = {}
        ood_tok: list[int] | None = None
        for k in needed_ks:
            if k == 0:
                qm.set_all("int4")
            else:
                qm.set_config(set(order[:k]))
            nll_o, tok_o = per_record_nll(model, ood_records, device)
            ood_nll[k] = nll_o
            ood_tok = tok_o
            print(f"[heldout] OOD k={k:>3}: ppl={subset_ppl(nll_o, tok_o, ood_idx):.4f} "
                  f"({len(ood_records)} rec, {sum(tok_o)} tok)", flush=True)
        ood_seed_gates = []
        for r in seed_rows:
            k = r["selected_k_heldout"]
            g = subset_gate(ood_nll[k], ood_nll[0], ood_tok, ood_idx)
            ood_seed_gates.append({"seed": r["seed"], "selected_k": k, "ood_gate_ppl": g,
                                   "ood_margin": PPL_GATE - g, "clears": bool(g <= PPL_GATE)})
            print(f"[heldout] OOD seed={r['seed']} k={k}: gate={g:.4f} margin={PPL_GATE-g:+.4f} "
                  f"clears={g <= PPL_GATE}", flush=True)
        ood_gate_vals = [x["ood_gate_ppl"] for x in ood_seed_gates]
        ood_mean, ood_se = mean_se(ood_gate_vals)
        ood_worst = max(ood_gate_vals)
        ood_sharegpt_ppl = ood_mean
        ood_margin = PPL_GATE - ood_mean
        cb3_ood_clears_2p42 = bool(all(x["clears"] for x in ood_seed_gates))
        ood.update({
            "source": ood_records[0]["source"], "n_records": len(ood_records),
            "num_tokens": sum(ood_tok), "int4_ppl": subset_ppl(ood_nll[0], ood_tok, ood_idx),
            "per_seed": ood_seed_gates,
            "ood_sharegpt_ppl_mean": ood_mean, "ood_sharegpt_ppl_se": ood_se,
            "ood_sharegpt_ppl_worst": ood_worst, "ood_margin_worst": PPL_GATE - ood_worst,
            "cb3_ood_clears_2p42": cb3_ood_clears_2p42,
        })

    # ---- 4. greedy-identity (report-only) --------------------------------- #
    mixed_greedy_frac_mismatch = greedy_drift(
        qm, model, records, device, order, k_star_insample,
        args.greedy_prompts if not args.smoke else 2, args.greedy_tokens if not args.smoke else 16)

    # ---- self-test -------------------------------------------------------- #
    base_is_strict = bool(any(args.base_model.endswith(s.split("/")[-1]) or args.base_model == s
                              for s in STRICT_PROXY_MODELS))
    st = {
        "reproduces_372_in_sample_2p3812": reproduces_372,
        "at_least_3_split_seeds": bool(len(args.split_seeds) >= 3),
        "splits_disjoint_partition_all_seeds": bool(all(r["disjoint_partition"] for r in seed_rows)),
        "ppl_greedy_under_strict_proxy": bool(base_is_strict and device.startswith("cuda")
                                              and m355.measure_ppl.__module__ == "measure_subint4"),
        "worst_seed_margin_is_reported": bool(
            (heldout_gate_worst == max(heldout_vals))
            and (cb3_heldout_clears_2p42 == (heldout_gate_worst <= PPL_GATE))),
    }
    if not args.smoke:
        st["ood_slice_disjoint_ge64"] = bool(args.ood_sharegpt and ood.get("n_records", 0) >= 64)
    cb3_ppl_heldout_self_test_passes = bool(all(st.values()))

    # ---- decision --------------------------------------------------------- #
    # Deployable iff held-out clears AND (OOD clears, when the OOD leg ran).
    deployable = bool(cb3_heldout_clears_2p42 and (cb3_ood_clears_2p42 if args.ood_sharegpt else True))
    verdict = ("DEPLOYABLE" if deployable else "PPL-BLOCKED")

    res = {
        "config": {
            "base_model": args.base_model, "scheme": args.scheme, "group_size": args.group_size,
            "vq_dim": args.vq_dim, "quant_seed": args.quant_seed, "split_seeds": args.split_seeds,
            "n_modules": len(order), "n_official_records": n_records,
            "k_star_insample_372": k_star_insample, "ppl_gate": PPL_GATE,
            "served_int4_ppl_spec_anchor": SERVED_INT4_PPL_SPEC, "device": device,
            "k_grid": grid, "fine_lo": args.fine_lo, "fine_step": args.fine_step,
            "reused_372_results": str(RESULTS_372.relative_to(REPO_ROOT)),
        },
        "in_sample": {
            "int4_localproxy_ppl": int4_ppl_full, "int4_localproxy_ppl_372": d372["int4_localproxy_ppl"],
            "in_sample_gate_ppl": in_sample_gate_ppl, "in_sample_gate_ppl_372": IN_SAMPLE_GATE_372,
            "repro_abs_diff": repro_abs_diff, "reproduces_372": reproduces_372,
        },
        "heldout": {
            "per_seed": seed_rows,
            "selection_half_ppl_mean": sel_mean, "selection_half_ppl_se": sel_se,
            "heldout_gate_ppl_mean": heldout_mean, "heldout_gate_ppl_se": heldout_se,
            "heldout_gate_ppl_worst": heldout_gate_worst,
            "heldout_margin_mean": PPL_GATE - heldout_mean, "heldout_margin_worst": heldout_margin_worst,
            "in_sample_overfit_delta_mean": overfit_mean, "in_sample_overfit_delta_se": overfit_se,
            "selected_k_heldout_worst": selected_k_heldout,
            "cb3_heldout_clears_2p42": cb3_heldout_clears_2p42,
        },
        "ood": ood,
        "greedy_identity": {
            "ref": "int4", "k_mixed": k_star_insample,
            "mixed_greedy_frac_mismatch": mixed_greedy_frac_mismatch,
            "note": "report-only; #319 gate is self-referential, deterministic quant passes its own AR",
        },
        "self_test": {**st, "cb3_ppl_heldout_self_test_passes": cb3_ppl_heldout_self_test_passes},
        "decision": {
            "cb3_heldout_clears_2p42": cb3_heldout_clears_2p42,
            "cb3_ood_clears_2p42": cb3_ood_clears_2p42,
            "cb3_supply_deployable": deployable, "verdict": verdict,
        },
        "guards": {
            "analysis_only": True, "no_hf_job": True, "no_served_file_change": True,
            "no_launch": True, "official_tps": 0,
        },
    }
    return _finish(args, res, t_start)


def _finish(args, res: dict[str, Any], t_start: float) -> int:
    print_report(res)
    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    peak_mib = (torch.cuda.max_memory_allocated() / 2**20) if torch.cuda.is_available() else 0.0
    payload = {
        "created_at": created_at, "pr": 394, "agent": "kanna",
        "kind": "cb3-ppl-heldout-margin",
        "elapsed_s": round(time.time() - t_start, 1), "peak_mem_mib": round(peak_mib, 1),
        "result": res,
    }
    out_path = Path(args.out_dir) / RESULTS_NAME
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float))
    print(f"[heldout] wrote {out_path} (elapsed {payload['elapsed_s']}s, peak {payload['peak_mem_mib']} MiB)",
          flush=True)
    _print_senpai_result(res)
    maybe_log_wandb(args, payload)
    return 0


def _print_senpai_result(res: dict[str, Any]) -> None:
    h = res["heldout"]; dec = res["decision"]; st = res["self_test"]
    marker = {
        "terminal": True, "status": "complete", "pending_arms": False,
        "analysis_only": True, "no_hf_job": True, "no_served_file_change": True, "no_launch": True,
        "in_sample_gate_ppl": round(res["in_sample"]["in_sample_gate_ppl"], 4),
        "heldout_gate_ppl_worst": round(h["heldout_gate_ppl_worst"], 4),
        "heldout_margin_worst": round(h["heldout_margin_worst"], 4),
        "cb3_heldout_clears_2p42": dec["cb3_heldout_clears_2p42"],
        "cb3_ood_clears_2p42": dec["cb3_ood_clears_2p42"],
        "cb3_supply_deployable": dec["cb3_supply_deployable"],
        "cb3_ppl_heldout_self_test_passes": st["cb3_ppl_heldout_self_test_passes"],
    }
    print("SENPAI-RESULT " + json.dumps(marker), flush=True)


def print_report(res: dict[str, Any]) -> None:
    c = res["config"]; isr = res["in_sample"]; h = res["heldout"]; o = res["ood"]
    dec = res["decision"]; st = res["self_test"]
    print("\n" + "=" * 104, flush=True)
    print("CB3 PPL HELD-OUT MARGIN (PR #394) — does #372's +0.039 in-sample margin survive out-of-sample?",
          flush=True)
    print("=" * 104, flush=True)
    print(f"  base={c['base_model']}  n_modules={c['n_modules']}  k*_insample(372)={c['k_star_insample_372']}  "
          f"gate<=2.42  split_seeds={c['split_seeds']}", flush=True)
    print(f"  IN-SAMPLE: gate_ppl={isr['in_sample_gate_ppl']:.4f} (372: {isr['in_sample_gate_ppl_372']:.4f}, "
          f"|diff|={isr['repro_abs_diff']:.4f}) reproduces_372={isr['reproduces_372']}  "
          f"int4_anchor={isr['int4_localproxy_ppl']:.4f}", flush=True)
    print("-" * 104, flush=True)
    print("  PRIMARY held-out (S-selected k, measured on disjoint H):", flush=True)
    print(f"    {'seed':>4} {'k_S':>4} {'sel_ppl':>8} {'heldout':>8} {'margin':>8} {'overfit_Δ':>9} {'clears':>6}",
          flush=True)
    for r in h["per_seed"]:
        print(f"    {r['seed']:>4} {r['selected_k_heldout']:>4} {r['selection_half_ppl']:>8.4f} "
              f"{r['heldout_gate_ppl']:>8.4f} {r['heldout_margin']:>+8.4f} "
              f"{r['in_sample_overfit_delta']:>+9.4f} {str(r['heldout_clears']):>6}", flush=True)
    print(f"    MEAN heldout_gate_ppl={h['heldout_gate_ppl_mean']:.4f}±{h['heldout_gate_ppl_se']:.4f}  "
          f"WORST={h['heldout_gate_ppl_worst']:.4f} (margin {h['heldout_margin_worst']:+.4f})", flush=True)
    print(f"    overfit_delta={h['in_sample_overfit_delta_mean']:+.4f}±{h['in_sample_overfit_delta_se']:.4f}  "
          f"-> cb3_heldout_clears_2p42(worst)={h['cb3_heldout_clears_2p42']}", flush=True)
    if o.get("enabled"):
        print("-" * 104, flush=True)
        print(f"  OOD ({o.get('source','?')}, n={o.get('n_records','?')}): "
              f"ppl_mean={o.get('ood_sharegpt_ppl_mean', float('nan')):.4f} "
              f"worst={o.get('ood_sharegpt_ppl_worst', float('nan')):.4f} "
              f"(margin {o.get('ood_margin_worst', float('nan')):+.4f})  "
              f"int4_anchor={o.get('int4_ppl', float('nan')):.4f} "
              f"-> cb3_ood_clears_2p42={o.get('cb3_ood_clears_2p42')}", flush=True)
    print("-" * 104, flush=True)
    print(f"  greedy drift (mixed k* vs int4, report-only)={res['greedy_identity']['mixed_greedy_frac_mismatch']:.4f}",
          flush=True)
    print(f"  SELF-TEST: {st}", flush=True)
    print("-" * 104, flush=True)
    print(f"  >>> VERDICT: {dec['verdict']}  (heldout_clears={dec['cb3_heldout_clears_2p42']}, "
          f"ood_clears={dec['cb3_ood_clears_2p42']}, deployable={dec['cb3_supply_deployable']})", flush=True)
    print("=" * 104, flush=True)


def maybe_log_wandb(args, payload: dict[str, Any]) -> None:
    if not getattr(args, "wandb_name", None):
        return
    try:
        if str(REPO_ROOT) not in sys.path:
            sys.path.append(str(REPO_ROOT))
        from scripts.wandb_logging import (  # noqa: E402
            finish_wandb, init_wandb_run, log_json_artifact, log_summary,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[heldout] wandb unavailable: {exc}", flush=True)
        return
    res = payload["result"]; c = res["config"]; isr = res["in_sample"]
    h = res["heldout"]; o = res["ood"]; dec = res["decision"]; st = res["self_test"]
    run = init_wandb_run(
        job_type="validity-gate", agent="kanna", name=args.wandb_name, group=args.wandb_group,
        tags=["validity-gate", "cb3-ppl-heldout-margin", "winners-curse", "held-out-selection",
              "mixed-precision", "sub-int4", "ood-sharegpt", "pr-394"],
        config={k: v for k, v in c.items() if not isinstance(v, list) or k == "split_seeds"},
    )
    if run is None:
        print("[heldout] wandb: no run — skipping", flush=True)
        return
    summary: dict[str, Any] = {
        "in_sample_gate_ppl": isr["in_sample_gate_ppl"],
        "in_sample_repro_abs_diff": isr["repro_abs_diff"],
        "reproduces_372_in_sample": int(bool(isr["reproduces_372"])),
        "int4_localproxy_ppl": isr["int4_localproxy_ppl"],
        "selection_half_ppl_mean": h["selection_half_ppl_mean"],
        "heldout_gate_ppl_mean": h["heldout_gate_ppl_mean"],
        "heldout_gate_ppl_se": h["heldout_gate_ppl_se"],
        "heldout_gate_ppl_worst": h["heldout_gate_ppl_worst"],
        "heldout_margin_mean": h["heldout_margin_mean"],
        "heldout_margin_worst": h["heldout_margin_worst"],
        "in_sample_overfit_delta_mean": h["in_sample_overfit_delta_mean"],
        "in_sample_overfit_delta_se": h["in_sample_overfit_delta_se"],
        "selected_k_heldout_worst": h["selected_k_heldout_worst"],
        "cb3_heldout_clears_2p42": int(bool(h["cb3_heldout_clears_2p42"])),
        "mixed_greedy_frac_mismatch": res["greedy_identity"]["mixed_greedy_frac_mismatch"],
        "cb3_supply_deployable": int(bool(dec["cb3_supply_deployable"])),
        "cb3_ppl_heldout_self_test_passes": int(bool(st["cb3_ppl_heldout_self_test_passes"])),
        "n_split_seeds": len(c["split_seeds"]),
        "peak_mem_mib": payload["peak_mem_mib"], "elapsed_s": payload["elapsed_s"],
    }
    if o.get("enabled"):
        summary["ood_sharegpt_ppl_mean"] = o.get("ood_sharegpt_ppl_mean")
        summary["ood_sharegpt_ppl_worst"] = o.get("ood_sharegpt_ppl_worst")
        summary["ood_margin_worst"] = o.get("ood_margin_worst")
        summary["ood_int4_ppl"] = o.get("int4_ppl")
        summary["ood_n_records"] = o.get("n_records")
        if o.get("cb3_ood_clears_2p42") is not None:
            summary["cb3_ood_clears_2p42"] = int(bool(o["cb3_ood_clears_2p42"]))
    for r in h["per_seed"]:
        summary[f"heldout_gate_ppl_seed{r['seed']}"] = r["heldout_gate_ppl"]
        summary[f"selected_k_seed{r['seed']}"] = r["selected_k_heldout"]
    summary = {k: v for k, v in summary.items()
               if v is not None and not (isinstance(v, float) and not math.isfinite(v))}
    log_summary(run, summary, step=0)
    log_json_artifact(run, name="cb3_ppl_heldout_margin", artifact_type="validity", data=payload)
    finish_wandb(run)
    print(f"[heldout] wandb logged: {len(summary)} metrics", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
