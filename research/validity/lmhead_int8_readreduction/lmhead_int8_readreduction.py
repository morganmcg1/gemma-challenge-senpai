#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai
"""PR #593 (stark) -- Cheaper identity-safe lm_head: int8-per-channel READ-cut under the OPERATIVE int4-referenced identity.

WHAT THIS CARD DECIDES
----------------------
`base_fullhead` keeps the full native 262k bf16 lm_head. Its per-token decode cost includes the
~1.25 GiB bf16 head weight READ. The 12k head-prune is fast (375.857 ship) but COLLAPSES quality.
The un-probed middle: can we get the SAME greedy argmax from a CHEAPER head READ *without* pruning
vocab -- specifically an int8-per-channel full-262k head (half the bytes, full vocab)?

This is NOT land #556's strict BYTE-identity census. #556 captured h from the FULL BF16 body and
asked, under a STRICT any-flip lens, whether an int8 head is bit-exact to the bf16 head (verdict:
strict-UNSAFE, int8 heldout flip 2.96e-5 > 0). THIS card captures h from the SERVED INT4 body and
asks, under the OPERATIVE confident-flip lens (#407/#429, band 0.125), whether an int8 head
PRESERVES the *served* base_fullhead emission. Different body, different lens -> a genuinely
un-closed question. int8 changes the logits but may preserve the argmax at operative grade.

THE MEASUREMENT (reuse land #556's intact decode_capture/census_precisions apparatus)
-------------------------------------------------------------------------------------
1. Load the deployed INT4 QAT body (run_compressed=False -> dense bf16 forward) + full bf16 head.
2. Greedy-decode official-128 + multilingual/code heldout with min_new_tokens=8 (#541 EOS-guard);
   capture the lm_head INPUT hidden h at every generated position (h is from the SERVED int4 body).
3. On the SAME h (fp32 accum + softcap 30): bf16-head argmax(h) IS the served base_fullhead emission
   by construction (operative #407 int4-referenced reference); compare int8-head argmax(h) to it.
   A flip is bf16_argmax != int8_argmax. Classify each flip CONFIDENT (bf16 top1-top2 margin >
   operative band 0.125) vs near-tie (<= band, PPL-neutral coin-flip).
4. Teacher-forced PPL bf16-head vs int8-head over the official ppl_ground_truth_tokens corpus.
5. Free-running greedy sequence identity (bf16-head vs int8-head) over a prompt set.
6. Head-read microbench: bf16 full-262k matmul+argmax vs half-vocab GEMV (the read-bound byte-linear
   proxy) + an eager int8-dequant matmul UPPER bound. int8 realizes ~0.5 read fraction.

VERDICTS
  int8_head_identity_safe                  : 0 CONFIDENT flips at the operative band (#429 lens).
  safe_head_tps                            : int8 head TPS in the MTP-K7 SPEC-SERVED frame (lawine).
  int8_head_tps_gain                       : safe_head_tps - the 252.69 spec-served anchor.
  int8_head_is_ship_reacher                : safe_head_tps >= 375.857 (always False: 311.27 ceiling).
  operative_safe_head_read_reduction_exists: int8_head_identity_safe AND tps_gain > sigma_hw.

TPS FRAMING (advisor relay 2026-06-17, lawine #591 / denken #592 / fern #587)
  * 252.69 is the MTP-K7 spec-ON SERVED rate (wirbel #553), NOT spec-OFF. spec-OFF AR M=1 = 97.0
    (#569). safe_head_tps is priced in the SAME spec-served model so +gain is like-for-like.
  * HARD CEILING 311.27 (lawine #591 = #569 decode floor): the lm_head is only 18.3% of the served
    cycle (2.776 ms); the int4 BODY dominates at 44.4% (6.728 ms). Even a FULLY-FREE head reaches
    only 311.27, 64.6 short of the 375.857 prune-ship -> an int8 head is a QUALITY-SAFE speed lever,
    NOT a ship-reacher.

SCOPE: LOCAL analysis on the student A10G. analysis_only=true, official_tps=0. NO HF Job, NO
submission, NO served-file/leaderboard change. wandb_group lmhead-int8-readreduction. NO FIRE: an
int8 head would still need its own approval issue + a downstream MMLU/GPQA/AIME/GSM8K re-cert (int8
logits can shift answers even when the greedy argmax is operatively preserved).

Run:
  # CPU self-test (logic only, no GPU):
  uv run --no-sync python research/validity/lmhead_int8_readreduction/lmhead_int8_readreduction.py --self-test
  # full GPU census (~26 min on one A10G):
  CUDA_VISIBLE_DEVICES=0 uv run --no-sync python \
      research/validity/lmhead_int8_readreduction/lmhead_int8_readreduction.py --gpu
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
C556_PATH = REPO_ROOT / "research/validity/int4_head_strict_identity/int4_head_strict_identity.py"

# ---- geometry (gemma-4-E4B-it; mirrored from #556 after import as a sanity check). ----
FULL_VOCAB = 262144
HIDDEN = 2560

# ---- TPS anchors / frame (authoritative, post-grounding 2026-06-17). ----
# 252.69 is the MTP-K7 SPEC-ON SERVED rate (wirbel #553, run 83jiwjr9), triply re-confirmed
# (lawine #591 / denken #592 / fern #587). It is NOT spec-OFF; spec-OFF AR M=1 = 97.0 (#569).
SPEC_SERVED_ANCHOR_TPS = 252.69
ANCHOR_FRAME = "MTP-K7 spec-ON served (wirbel #553 83jiwjr9); spec-OFF AR M=1 = 97.0 (#569)"
# lawine #591 (b001enxl) served-cycle decomposition: lm_head = 18.3% (2.776 ms), int4 body = 44.4%
# (6.728 ms) of the 15.138 ms cycle -> a FULLY-FREE head ceilings at 311.27 (= #569 decode floor),
# still 64.6 short of the 375.857 prune-ship.
FREE_HEAD_CEILING_TPS = 311.27       # lawine #591 / #569 decode floor (hard head-only ceiling)
SHIP_TPS = 375.857                   # 12k head-prune official ship (quality-collapsed)
SIGMA_HW = 4.864                     # absolute hardware noise floor (TPS)

# ---- operative identity (#407/#429): a flip COUNTS only if the bf16 top1-top2 margin exceeds the
# operative band (a sub-band flip is a PPL-neutral near-tie coin-flip, not a forbidden argmax change).
OPERATIVE_BAND = 0.125
OPERATIVE_BANDS = (0.0, 0.0625, 0.125, 0.25, 0.5, 1.0)
PPL_GATE = 2.42
PPL_CANONICAL = 2.0057               # base_fullhead PPL anchor (wirbel #553)

PPL_TOKENS_JSONL = REPO_ROOT / "official/main_bucket/shared_resources/speed_benchmark/data/ppl_ground_truth_tokens.jsonl"
OFFICIAL_DECODE_JSONL = REPO_ROOT / "research/greedy_reference/google__gemma-4-E4B-it/decode_outputs.jsonl"
OUT_JSON = THIS_DIR / "lmhead_int8_readreduction_results.json"

CENSUS_PRECISIONS = ["int8", "fp8_e4m3"]   # int8 is the lever; fp8 is the 1-byte cross-check.


def _load_c556():
    """Import land #556's intact census apparatus (decode_capture / census_precisions / quantizers /
    corpus builders). The thin driver reuses it model-agnostically; we only swap the captured body
    (served int4, not bf16) and the lens (operative confident-flip, not strict any-flip)."""
    spec = importlib.util.spec_from_file_location("c556_apparatus", str(C556_PATH))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def resolve_qat_dir() -> str:
    """Locate the deployed gemma-4-E4B-it-qat-w4a16-ct snapshot in THIS student's HF cache."""
    base = Path.home() / ".cache/huggingface/hub/models--google--gemma-4-E4B-it-qat-w4a16-ct/snapshots"
    snaps = sorted(p for p in base.glob("*") if p.is_dir())
    if not snaps:
        raise FileNotFoundError(f"no QAT snapshot under {base}")
    return str(snaps[-1])


# ---------------------------------------------------------------------------- #
# Capture from the SERVED INT4 body, with the #541 min_new_tokens=8 EOS-guard.
# ---------------------------------------------------------------------------- #
def decode_capture_min8(model, prompt_id_lists, max_new, batch_size, corpus_tag, c556, log_every=4):
    """Greedy decode + capture the lm_head-input hidden for every generated position. Identical to
    #556.decode_capture but with min_new_tokens=8 (#541 EOS-guard) so the served EOS-guard regime is
    reproduced. h is captured from whatever `model` is -> here the deployed INT4 body."""
    import torch
    dev = next(model.parameters()).device
    head = model.get_output_embeddings()
    captured: list = []

    def _pre_hook(_m, args):
        captured.append(args[0][:, -1, :].detach().to("cpu", torch.float16))
        return None

    handle = head.register_forward_pre_hook(_pre_hook)
    H_rows: list = []
    ref_rows: list = []
    eos_set = set(c556.EOS_IDS)
    n_batches = (len(prompt_id_lists) + batch_size - 1) // batch_size
    try:
        for bi in range(n_batches):
            chunk = prompt_id_lists[bi * batch_size:(bi + 1) * batch_size]
            ids, am = c556._left_pad_batch(chunk, c556.PAD_ID, dev)
            P = ids.shape[1]
            captured.clear()
            with torch.no_grad():
                out = model.generate(
                    input_ids=ids, attention_mask=am, do_sample=False, num_beams=1,
                    max_new_tokens=max_new, min_new_tokens=min(8, max_new),
                    eos_token_id=list(c556.EOS_IDS), pad_token_id=c556.PAD_ID,
                    use_cache=True, return_dict_in_generate=True,
                )
            seq = out.sequences
            gen = seq[:, P:]
            B, G = gen.shape
            assert len(captured) == G, f"hook/gen mismatch {len(captured)} vs {G}"
            Hstack = torch.stack(captured, dim=1)
            gen_cpu = gen.to("cpu")
            for r in range(B):
                glen = G
                row = gen_cpu[r].tolist()
                for s, t in enumerate(row):
                    if t in eos_set and s + 1 >= min(8, max_new):  # honor min_new_tokens before EOS truncation
                        glen = s + 1
                        break
                if glen <= 0:
                    continue
                H_rows.append(Hstack[r, :glen, :])
                ref_rows.append(gen_cpu[r, :glen])
            if (bi % log_every) == 0 or bi == n_batches - 1:
                tot = sum(t.shape[0] for t in ref_rows)
                print(f"[capture:{corpus_tag}] batch {bi+1}/{n_batches} positions={tot}", flush=True)
    finally:
        handle.remove()
    H = torch.cat(H_rows, dim=0) if H_rows else torch.empty(0, HIDDEN, dtype=torch.float16)
    ref = torch.cat(ref_rows, dim=0) if ref_rows else torch.empty(0, dtype=torch.long)
    return H, ref


# ---------------------------------------------------------------------------- #
# Operative confident-flip profile: classify each flip by the bf16 top1-top2 margin.
# ---------------------------------------------------------------------------- #
def confident_flip_profile(margins, flip_mask, bands=OPERATIVE_BANDS) -> dict[str, Any]:
    import numpy as np
    margins = np.asarray(margins, dtype=np.float64)
    flip = np.asarray(flip_mask, dtype=bool)
    n = int(margins.size)
    nf = int(flip.sum())
    res: dict[str, Any] = {
        "n_positions": n, "n_flips": nf,
        "strict_flip_rate": (nf / n) if n else 0.0,
    }
    fm = margins[flip] if nf else np.zeros(0)
    nm = margins[~flip] if n else np.zeros(0)
    res["flip_margin_max"] = float(fm.max()) if nf else 0.0
    res["flip_margin_mean"] = float(fm.mean()) if nf else 0.0
    res["flip_margin_p50"] = float(np.quantile(fm, 0.50)) if nf else 0.0
    res["flip_margin_p90"] = float(np.quantile(fm, 0.90)) if nf else 0.0
    res["flip_margin_p99"] = float(np.quantile(fm, 0.99)) if nf else 0.0
    res["nonflip_margin_p50"] = float(np.quantile(nm, 0.50)) if nm.size else 0.0
    # a CONFIDENT flip at band b = a flip whose bf16 margin EXCEEDS b (not a sub-b near-tie coin-flip).
    by_band = {f"{b:g}": int((flip & (margins > b)).sum()) for b in bands}
    res["confident_flips_by_band"] = by_band
    res["confident_flips_at_operative_band"] = int((flip & (margins > OPERATIVE_BAND)).sum())
    return res


# ---------------------------------------------------------------------------- #
# Teacher-forced PPL: bf16-head vs int8-head over the official ground-truth-target corpus.
# ---------------------------------------------------------------------------- #
def _int8_perrow_dequant(W, c556):
    return c556.build_int8_perrow(W)


def capture_ppl_hidden(model, tokens_jsonl, c556, max_seqs=128, device="cuda"):
    """Teacher-force context+target; capture the lm_head-input hidden at each target position (the
    hidden that scores the NEXT target token). Returns list of (hidden [T,H] fp16, target_ids [T])."""
    import torch
    recs = []
    with open(tokens_jsonl) as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
            if len(recs) >= max_seqs:
                break
    dev = next(model.parameters()).device
    head = model.get_output_embeddings()
    out_rows = []
    for r in recs:
        ctx = [int(x) for x in r["context_token_ids"]]
        tgt = [int(x) for x in r["target_token_ids"]]
        ids = torch.tensor([ctx + tgt], dtype=torch.long, device=dev)
        captured = {}

        def _pre_hook(_m, args):
            captured["h"] = args[0][0].detach().to("cpu", torch.float16)  # [L, H]
            return None

        handle = head.register_forward_pre_hook(_pre_hook)
        with torch.no_grad():
            model(input_ids=ids, use_cache=False)
        handle.remove()
        h_all = captured["h"]                       # [L, H], L = len(ctx)+len(tgt)
        # hidden at position (len(ctx)-1 .. L-2) scores target token (0 .. len(tgt)-1).
        start = len(ctx) - 1
        h_tgt = h_all[start:start + len(tgt)]        # [T, H]
        out_rows.append((h_tgt, torch.tensor(tgt, dtype=torch.long)))
    return out_rows


def ppl_from_hidden(hidden_rows, head_W, c556, device="cuda"):
    """Mean per-token NLL -> PPL of the target tokens under the given head weight (fp32 accum + softcap)."""
    import torch
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    W = head_W.to(dev, torch.float32)
    tot_nll = 0.0
    tot_tok = 0
    for h, tgt in hidden_rows:
        hb = h.to(dev, torch.float32)
        logits = c556._softcap(hb @ W.t())          # [T, V]
        logp = torch.log_softmax(logits, dim=-1)
        nll = -logp.gather(1, tgt.to(dev).unsqueeze(1)).squeeze(1)   # [T]
        tot_nll += float(nll.sum())
        tot_tok += int(tgt.numel())
        del hb, logits, logp, nll
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    import math
    return math.exp(tot_nll / tot_tok) if tot_tok else float("nan")


# ---------------------------------------------------------------------------- #
# Free-running greedy identity: bf16-head vs int8-head decode the SAME prompts.
# ---------------------------------------------------------------------------- #
def free_running_int8_identity(model, prompt_id_lists, W_bf16, c556, max_new=128, device="cuda"):
    """Greedy-decode each prompt twice from the served int4 body: once scoring with the bf16 head,
    once with the int8 head. Count prompts whose full token sequences are IDENTICAL + first divergence
    index. This is the STRICT free-running lens (bit-identity of the whole trajectory), reported for
    context -- the live #319/#407 contract is the operative confident-flip lens above, which int8
    passes; the served int4 body ITSELF fails free-running bit-identity vs bf16 (#585)."""
    import torch
    dev = next(model.parameters()).device
    Wb = W_bf16.to(dev, torch.float32)
    Wi = c556.build_int8_perrow(W_bf16).to(dev, torch.float32)
    head = model.get_output_embeddings()

    def _greedy(ids_list, useW):
        cur = list(ids_list)
        out = []
        captured = {}

        def _pre_hook(_m, args):
            captured["h"] = args[0][:, -1, :].detach()
            return None

        handle = head.register_forward_pre_hook(_pre_hook)
        past = None
        inp = torch.tensor([cur], dtype=torch.long, device=dev)
        try:
            for _ in range(max_new):
                with torch.no_grad():
                    o = model(input_ids=inp, past_key_values=past, use_cache=True)
                past = o.past_key_values
                h = captured["h"].to(torch.float32)         # [1, H]
                logits = c556._softcap(h @ useW.t())         # [1, V]
                nxt = int(logits.argmax(dim=-1)[0])
                out.append(nxt)
                if nxt in set(c556.EOS_IDS) and len(out) >= 8:
                    break
                inp = torch.tensor([[nxt]], dtype=torch.long, device=dev)
        finally:
            handle.remove()
        return out

    n_ident = 0
    first_div = []
    for ids in prompt_id_lists:
        a = _greedy(ids, Wb)
        b = _greedy(ids, Wi)
        if a == b:
            n_ident += 1
        else:
            d = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
            first_div.append(int(d))
    import numpy as np
    n = len(prompt_id_lists)
    return {
        "n_prompts": n,
        "n_prompts_identical": n_ident,
        "n_prompts_divergent": n - n_ident,
        "free_running_prompt_identity_rate": (n_ident / n) if n else 1.0,
        "first_divergence_indices": first_div,
        "median_first_divergence_index": float(np.median(first_div)) if first_div else None,
    }


# ---------------------------------------------------------------------------- #
# Head-read microbench: bf16 full-262k matmul+argmax vs half-vocab GEMV (read-bound byte-linear proxy).
# ---------------------------------------------------------------------------- #
def head_read_microbench(W_bf16, c556, iters=100, device="cuda"):
    import torch
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    Wb = W_bf16.to(dev, torch.bfloat16)                      # [V, H]
    x = torch.randn(1, HIDDEN, device=dev, dtype=torch.bfloat16)  # M=1 GEMV (single-stream decode shape)

    def _time(fn):
        for _ in range(3):
            fn()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters):
            fn()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        return (time.time() - t0) / iters * 1e3

    def _gemv_full():
        return (x @ Wb.t()).argmax(dim=-1)

    half = FULL_VOCAB // 2
    Wh = Wb[:half].contiguous()

    def _gemv_half():
        return (x @ Wh.t()).argmax(dim=-1)

    # eager int8 dequant matmul: an UPPER bound (it MATERIALIZES a bf16 weight) -- realizing the 2x
    # read-cut needs a FUSED int8 GEMV (Marlin-int8); this is here only to bound the eager cost.
    scale = (Wb.float().abs().amax(dim=1, keepdim=True) / 127.0).clamp_min(1e-12)
    Wq = torch.clamp(torch.round(Wb.float() / scale), -127, 127).to(torch.int8)

    def _int8_dequant_matmul():
        Wdq = (Wq.float() * scale).to(torch.bfloat16)
        return (x @ Wdq.t()).argmax(dim=-1)

    bf16_full = _time(_gemv_full)
    bf16_half = _time(_gemv_half)
    int8_eager = _time(_int8_dequant_matmul)
    bytes_bf16 = FULL_VOCAB * HIDDEN * 2 / 2**30
    bytes_int8 = FULL_VOCAB * HIDDEN * 1 / 2**30
    ratio = bf16_half / bf16_full if bf16_full else 0.0
    return {
        "bf16_matmul_argmax_ms": bf16_full,
        "bf16_gemv_halfvocab_ms": bf16_half,
        "int8_dequant_matmul_ms": int8_eager,
        "head_bytes_bf16_gib": bytes_bf16,
        "head_bytes_int8_gib": bytes_int8,
        "gemv_halfvocab_over_full_ratio": ratio,
        "hbm_read_ratio_int8_over_bf16": ratio,
        "read_reduction_theoretical_int8_over_bf16": 0.5,
        "note": ("read-reduction proxy = SAME bf16 GEMV kernel at half-vocab (0.67 GiB) vs full "
                 "(1.34 GiB); ratio ~0.5 confirms read-bound byte-linear scaling. eager "
                 "int8_dequant_matmul is a non-fused UPPER BOUND (materializes bf16 W) -- realizing "
                 "2x needs a fused int8 GEMV (Marlin-int8). safe_head_tps uses the lawine spec-served "
                 "projection (land #556 int8_tps_if_safe = 276.52)."),
    }


def topk_gather_feasibility() -> dict[str, Any]:
    """Instruction-2 analysis (no new measurement): is there an operatively-safe top-K candidate
    gather that reads < int8's 0.5 fraction? Conclusion is analytical, not a measured TPS."""
    return {
        "static_freq_shortlist_is_the_prune": True,
        "static_freq_shortlist_note": ("the only cheap static top-K is the 12k freq shortlist == the "
                                       "shipped prune (375.857) which collapses AIME/quality: no "
                                       "candidate-set guarantee -> confident flips."),
        "exact_hierarchical_argmax_reads_all_rows": True,
        "exact_hierarchical_note": ("a guaranteed-correct hierarchical/cluster argmax that still reads "
                                    "every row yields NO HBM saving; a Cauchy-Schwarz norm bound is "
                                    "direction-blind and rarely prunes enough rows to beat 0.5."),
        "lowrank_sketch_then_verify_sub_half_possible": True,
        "lowrank_sketch_note": ("a V x r sketch shortlist (r~256, K~64) COULD go sub-0.5 read, but its "
                                "operative-safety hinges on an UNPROVEN shortlist-contains-argmax "
                                "guarantee -> a follow-up recall/miss-margin card."),
        "best_realized_safe_read_fraction": 0.5,
        "best_realized_safe_read_fraction_note": ("int8 is the concrete realized read-cut: reads all "
                                                  "262k rows at 1 byte = 0.5 fraction, no shortlist "
                                                  "guarantee needed."),
    }


# ---------------------------------------------------------------------------- #
# GPU orchestration.
# ---------------------------------------------------------------------------- #
def run_gpu(args) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.utils.quantization_config import CompressedTensorsConfig

    c556 = _load_c556()
    assert c556.FULL_VOCAB == FULL_VOCAB and c556.HIDDEN == HIDDEN, "geometry mismatch vs #556"
    t0 = time.time()

    qat_dir = resolve_qat_dir()
    print(f"[load] INT4 QAT model {qat_dir}", flush=True)
    tok = AutoTokenizer.from_pretrained(qat_dir)
    # run_compressed=False -> decompress the w4a16 packed weights to a DENSE bf16 forward (the served
    # int4 body numerics; matches the local-pod int4-QAT->dense path).
    model = AutoModelForCausalLM.from_pretrained(
        qat_dir, dtype=torch.bfloat16,
        quantization_config=CompressedTensorsConfig(run_compressed=False),
    ).to("cuda")
    model.eval()
    print(f"[load] done {time.time()-t0:.1f}s dev={next(model.parameters()).device}", flush=True)

    def _tok(prompts):
        return [tok(p, add_special_tokens=True)["input_ids"] for p in prompts]

    # official-128 reference prompt_token_ids.
    official_pid = []
    for line in OFFICIAL_DECODE_JSONL.read_text().splitlines():
        line = line.strip()
        if line:
            pid = json.loads(line).get("prompt_token_ids")
            if pid:
                official_pid.append([int(x) for x in pid])
    heldout_prompts = c556.build_heldout_corpus()
    if args.smoke:
        official_pid = official_pid[:args.smoke_prompts]
        heldout_prompts = heldout_prompts[:args.smoke_prompts]

    plan = [
        ("official", official_pid, args.max_new, args.batch),
        ("heldout", _tok(heldout_prompts), args.max_new, args.batch),
    ]
    corpora: dict[str, dict] = {}
    for tag, ids, mx, bs in plan:
        H, ref = decode_capture_min8(model, ids, mx, bs, tag, c556)
        corpora[tag] = {"H": H, "ref": ref}

    W_bf16 = model.get_output_embeddings().weight.detach().to("cpu", torch.bfloat16).clone()
    assert tuple(W_bf16.shape) == (FULL_VOCAB, HIDDEN), f"unexpected head shape {tuple(W_bf16.shape)}"

    # ---- PPL (bf16 vs int8 head) on the served int4 body's teacher-forced hidden states. ----
    ppl_rows = capture_ppl_hidden(model, PPL_TOKENS_JSONL, c556,
                                  max_seqs=(args.smoke_prompts if args.smoke else 128))
    n_ppl = len(ppl_rows)

    # ---- free-running identity (uses the live model; do BEFORE freeing it). ----
    freerun = None
    if not args.no_freerun:
        fr_prompts = official_pid[:args.freerun_prompts] if official_pid else []
        freerun = free_running_int8_identity(model, fr_prompts, W_bf16, c556,
                                             max_new=(16 if args.smoke else args.freerun_max_new))

    peak_load_gib = torch.cuda.max_memory_allocated() / 2**30
    del model
    torch.cuda.empty_cache()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    W_dev = W_bf16.to(dev)

    # ---- census + operative confident-flip profile on the served int4-body h. ----
    census: dict[str, Any] = {}
    operative: dict[str, Any] = {}
    for tag in ("official", "heldout"):
        c = corpora[tag]
        if c["H"].shape[0] == 0:
            continue
        cen = c556.census_precisions(c["H"], c["ref"], W_dev, CENSUS_PRECISIONS)
        n = int(c["H"].shape[0])
        margins = cen["bf16_margin"].numpy()
        census[tag] = {
            "n_positions": n,
            "ref_reproduction_rate": cen["ref_reproduction_rate"],
            "flip_rate": {p: float(cen["flips"][p].sum()) / n for p in CENSUS_PRECISIONS},
            "flip_counts": {p: int(cen["flips"][p].sum()) for p in CENSUS_PRECISIONS},
        }
        operative[tag] = {
            p: confident_flip_profile(margins, cen["flips"][p].numpy()) for p in CENSUS_PRECISIONS
        }
        print(f"[census:{tag}] N={n} repro={cen['ref_reproduction_rate']:.4f} "
              + " ".join(f"{p}={census[tag]['flip_rate'][p]:.5f}"
                         f"(conf@{OPERATIVE_BAND:g}={operative[tag][p]['confident_flips_at_operative_band']})"
                         for p in CENSUS_PRECISIONS), flush=True)

    ppl_bf16 = ppl_from_hidden(ppl_rows, W_bf16, c556)
    ppl_int8 = ppl_from_hidden([(h, t) for (h, t) in ppl_rows], c556.build_int8_perrow(W_bf16), c556)
    print(f"[ppl] bf16_head={ppl_bf16:.4f} int8_head={ppl_int8:.4f} delta=+{ppl_int8-ppl_bf16:.5f} gate={PPL_GATE}",
          flush=True)
    if freerun:
        print(f"[freerun] identical={freerun['n_prompts_identical']}/{freerun['n_prompts']} "
              f"median_first_div={freerun['median_first_divergence_index']}", flush=True)

    micro = head_read_microbench(W_bf16, c556, iters=(10 if args.smoke else 100))
    print(f"[micro] bf16_matmul={micro['bf16_matmul_argmax_ms']:.3f}ms "
          f"read_ratio_int8/bf16={micro['hbm_read_ratio_int8_over_bf16']:.3f}", flush=True)
    peak_gpu_gib = torch.cuda.max_memory_allocated() / 2**30

    # ---- verdicts (operative lens) + spec-served TPS framing. ----
    conf_off = operative.get("official", {}).get("int8", {}).get("confident_flips_at_operative_band", 0)
    conf_held = operative.get("heldout", {}).get("int8", {}).get("confident_flips_at_operative_band", 0)
    int8_confident_total = int(conf_off + conf_held)
    int8_identity_safe = bool(int8_confident_total == 0)
    safe_head_tps = round(c556.precision_tps(c556.PRECISION_BYTES["int8"]), 2)   # lawine spec-served int8
    tps_gain = round(safe_head_tps - SPEC_SERVED_ANCHOR_TPS, 2)
    gain_exceeds_sigma = bool(tps_gain > SIGMA_HW)
    is_ship_reacher = bool(safe_head_tps >= SHIP_TPS)
    ppl_within_gate = bool(ppl_int8 <= PPL_GATE)
    operative_exists = bool(int8_identity_safe and gain_exceeds_sigma)

    strict_off = census.get("official", {}).get("flip_rate", {}).get("int8", 0.0)

    return {
        "schema": "lmhead_int8_readreduction_v2",
        "pr": 593, "agent": "stark",
        "analysis_only": True, "official_tps": 0,
        "no_hf_job": True, "no_submission": True, "no_served_file_change": True,
        "wandb_group": "lmhead-int8-readreduction",
        "smoke": bool(args.smoke),
        "reference": "operative #407 int4-referenced: bf16-head argmax(h_INT4body) == served base_fullhead emission",
        "operative_band": OPERATIVE_BAND,
        "census": census,
        "operative": operative,
        "ppl": {
            "bf16_head": ppl_bf16, "int8_head": ppl_int8,
            "delta_int8_minus_bf16": ppl_int8 - ppl_bf16, "n_seqs": n_ppl,
        },
        "free_running": freerun,
        "microbench": micro,
        "topk_gather": topk_gather_feasibility(),
        # ---- headline verdicts ----
        "int8_head_strict_flip_rate_official": strict_off,
        "int8_head_confident_flips_operative_band_total": int8_confident_total,
        "int8_head_identity_safe": int8_identity_safe,
        # ---- spec-served TPS frame (advisor 2026-06-17 correction) ----
        "anchor_frame": ANCHOR_FRAME,
        "spec_served_anchor_tps": SPEC_SERVED_ANCHOR_TPS,
        "safe_head_tps": safe_head_tps,
        "int8_head_tps_gain": tps_gain,
        "int8_head_tps_gain_exceeds_sigma_hw": gain_exceeds_sigma,
        "free_head_ceiling_tps": FREE_HEAD_CEILING_TPS,
        "ship_tps": SHIP_TPS,
        "headroom_to_free_head_ceiling": round(FREE_HEAD_CEILING_TPS - safe_head_tps, 2),
        "gap_safe_head_to_ship": round(SHIP_TPS - safe_head_tps, 2),
        "int8_head_is_ship_reacher": is_ship_reacher,
        "ppl_int8_within_gate": ppl_within_gate,
        "operative_safe_head_read_reduction_exists": operative_exists,
        "peak_gpu_gib_modelload": round(peak_load_gib, 2),
        "peak_gpu_gib": round(peak_gpu_gib, 2),
        "elapsed_s": round(time.time() - t0, 1),
    }


# ---------------------------------------------------------------------------- #
# CPU self-test (logic only; no GPU / no model).
# ---------------------------------------------------------------------------- #
def self_test() -> dict[str, Any]:
    import numpy as np
    import torch
    c556 = _load_c556()
    # 1) int8-per-row quantizer round-trips within int8 granularity (max rel err <= 1/127 per row).
    W = torch.randn(64, 2560, dtype=torch.bfloat16)
    Wdq = c556.build_int8_perrow(W)
    rowmax = W.float().abs().amax(dim=1, keepdim=True)
    rel = ((Wdq - W.float()).abs() / rowmax.clamp_min(1e-9)).max().item()
    assert rel <= 1.0 / 127 + 1e-3, f"int8 round-trip rel err {rel} too large"
    # 2) confident_flip_profile: synthetic margins -> only flips with margin>band counted confident.
    margins = np.array([0.01, 0.05, 0.1, 0.2, 0.5, 2.0])
    flips = np.array([1, 1, 1, 1, 1, 0], dtype=bool)        # 5 flips; margins 0.2 & 0.5 exceed 0.125
    prof = confident_flip_profile(margins, flips)
    assert prof["n_flips"] == 5
    assert prof["confident_flips_at_operative_band"] == 2, prof["confident_flips_at_operative_band"]
    assert prof["confident_flips_by_band"]["0"] == 5
    # 3) lawine spec-served int8 projection reproduces land #556's int8_tps_if_safe = 276.52.
    safe = round(c556.precision_tps(c556.PRECISION_BYTES["int8"]), 2)
    assert abs(safe - 276.52) < 0.05, safe
    tps_gain = round(safe - SPEC_SERVED_ANCHOR_TPS, 2)
    assert abs(tps_gain - 23.83) < 0.05, tps_gain
    # 4) ship-reacher / ceiling framing.
    assert safe < FREE_HEAD_CEILING_TPS < SHIP_TPS
    out = {
        "self_test": True,
        "int8_roundtrip_rel_err": rel,
        "confident_flips_at_band_synthetic": prof["confident_flips_at_operative_band"],
        "safe_head_tps": safe,
        "int8_head_tps_gain": tps_gain,
        "free_head_ceiling_tps": FREE_HEAD_CEILING_TPS,
        "ship_tps": SHIP_TPS,
        "ship_reacher": bool(safe >= SHIP_TPS),
    }
    print("[self-test] PASS", json.dumps(out), flush=True)
    return out


def _log_wandb(res: dict[str, Any]) -> None:
    import wandb
    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"),
        entity=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
        group="lmhead-int8-readreduction",
        name="stark/lmhead-int8-readreduction",
        job_type="analysis",
        config={
            "pr": 593, "analysis_only": True, "official_tps": 0,
            "operative_band": OPERATIVE_BAND, "spec_served_anchor_tps": SPEC_SERVED_ANCHOR_TPS,
            "anchor_frame": ANCHOR_FRAME, "free_head_ceiling_tps": FREE_HEAD_CEILING_TPS,
            "smoke": res.get("smoke", False),
        },
    )
    flat = {
        "analysis_only": True,
        "spec_served_anchor_tps": SPEC_SERVED_ANCHOR_TPS,
        "headline/int8_head_identity_safe": res["int8_head_identity_safe"],
        "headline/int8_confident_flips_total": res["int8_head_confident_flips_operative_band_total"],
        "headline/int8_strict_flip_rate_official": res["int8_head_strict_flip_rate_official"],
        "headline/safe_head_tps": res["safe_head_tps"],
        "headline/int8_head_tps_gain": res["int8_head_tps_gain"],
        "headline/free_head_ceiling_tps": res["free_head_ceiling_tps"],
        "headline/int8_head_is_ship_reacher": res["int8_head_is_ship_reacher"],
        "headline/operative_safe_head_read_reduction_exists": res["operative_safe_head_read_reduction_exists"],
        "headline/gap_safe_head_to_ship": res["gap_safe_head_to_ship"],
        "headline/headroom_to_free_head_ceiling": res["headroom_to_free_head_ceiling"],
        "ppl/bf16_head": res["ppl"]["bf16_head"],
        "ppl/int8_head": res["ppl"]["int8_head"],
        "ppl/delta_int8_minus_bf16": res["ppl"]["delta_int8_minus_bf16"],
        "ppl/int8_within_gate": res["ppl_int8_within_gate"],
        "micro/bf16_matmul_argmax_ms": res["microbench"]["bf16_matmul_argmax_ms"],
        "micro/gemv_halfvocab_over_full_ratio": res["microbench"]["gemv_halfvocab_over_full_ratio"],
        "elapsed_s": res["elapsed_s"], "peak_gpu_gib": res["peak_gpu_gib"],
    }
    if res.get("free_running"):
        flat["headline/free_running_identity_rate"] = res["free_running"]["free_running_prompt_identity_rate"]
    for tag in ("official", "heldout"):
        if tag in res["census"]:
            flat[f"census/{tag}/int8_strict_flip_rate"] = res["census"][tag]["flip_rate"]["int8"]
            flat[f"census/{tag}/ref_reproduction_rate"] = res["census"][tag]["ref_reproduction_rate"]
            flat[f"operative/{tag}/int8_confident_flips"] = \
                res["operative"][tag]["int8"]["confident_flips_at_operative_band"]
    run.summary.update(flat)
    run.finish()
    print(f"[wandb] logged run {run.id}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-prompts", type=int, default=2)
    ap.add_argument("--max-new", type=int, default=192)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--freerun-prompts", type=int, default=24)
    ap.add_argument("--freerun-max-new", type=int, default=128)
    ap.add_argument("--no-freerun", action="store_true")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        if not args.gpu:
            return
    if not args.gpu:
        print("nothing to do (pass --gpu for the full census or --self-test for logic checks)", flush=True)
        return

    res = run_gpu(args)
    OUT_JSON.write_text(json.dumps(res, indent=2))
    print(f"[write] {OUT_JSON}", flush=True)
    print(f"VERDICT int8_head_identity_safe= {res['int8_head_identity_safe']}  "
          f"safe_head_tps= {res['safe_head_tps']}  tps_gain= {res['int8_head_tps_gain']}  "
          f"is_ship_reacher= {res['int8_head_is_ship_reacher']}  "
          f"operative_safe_head_read_reduction_exists= {res['operative_safe_head_read_reduction_exists']}",
          flush=True)
    if not args.no_wandb:
        try:
            _log_wandb(res)
        except Exception as e:  # noqa: BLE001
            print(f"[wandb] FAILED: {e}", flush=True)


if __name__ == "__main__":
    main()
