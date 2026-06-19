#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""PR #748 (land) -- benign-tie classification of the BI=1 served spec divergences.

The advisor (PR #748 comment, 2026-06-19T17:39Z, relaying lawine #752) instructs that a
plain-BI=1 served non-128/128 must be read as the EXPECTED adaptive-split-KV ULP-tie cascade
and classified `self-consistent-not-byteexact` (check tau=0.3 + PPL the way lawine did), NOT
a transfer failure. This script supplies exactly those two checks, offline (enforce_eager),
replicating land #720's gap_probe protocol and the official PPL metric (ppl_endpoint.py).

It does NOT generate -- it SCORES already-captured token streams:

(1) tau=0.3 self-consistency gap_probe (#720 `gap_probe.json` protocol). For every prompt
    where the BI=1 spec stream != BI=1 AR stream, take the first-divergence index d. The
    shared prefix (prompt + completion[:d]) is identical in both arms. Re-score it through the
    model and read the top-1/top-2 next-token logprobs; gap_nat = top1 - top2 (NATS). A flip is
    a "confident genuine flip" iff gap_nat >= tau (0.3). #720 got 0 confident flips, max_gap
    0.25 nat (gaps quantize to {0, 0.125, 0.25} -- bf16 ULP steps). 0 confident => the
    divergences are don't-care numeric ties (quality-neutral), i.e. self-consistent.

(2) PPL (official ppl_endpoint.py math, offline): token-weighted exp(sum NLL / sum tokens) over
    the 128 ppl_ground_truth_tokens.jsonl context+target records. Run under BI=1 and (separately)
    BI=0 to show the reduction-order change is quality-neutral; absolute PPL is on the loadable
    full-vocab QAT proxy (google/gemma-4-E4B-it-qat-w4a16-ct), NOT the deployed pruned-16k head,
    so it anchors near -- not exactly at -- the deployed 2.019.

LOCAL A10G only. analysis_only -- NO HF Job, NO submission, NO served-file change. Scoring is
enforce_eager (canonical), matching #720; a tie under one reduction order is a tie under the
other (the whole point of tau), so eager scoring faithfully classifies the served divergences.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[3]
PPL_DATA = (ROOT / "official/main_bucket/shared_resources/speed_benchmark/data/"
            "ppl_ground_truth_tokens.jsonl")
MODEL_ID = "google/gemma-4-E4B-it-qat-w4a16-ct"
TAU = 0.3  # nats; #720 gap_probe threshold


def lp_value(entry) -> float:
    """vLLM Logprob -> float (.logprob), tolerant of a bare float."""
    if entry is None:
        return float("-inf")
    if isinstance(entry, (int, float)):
        return float(entry)
    lp = getattr(entry, "logprob", None)
    return float(lp) if lp is not None else float("-inf")


# ---- (2) PPL: official ppl_endpoint.py math, offline ----


def normalized_ppl_record(record: dict, index: int) -> dict:
    rid = str(record.get("id", index))
    if "context_token_ids" in record and "target_token_ids" in record:
        context = record["context_token_ids"]
        target = record["target_token_ids"]
        ptoks = list(context) + list(target)
        score_start, score_end = len(context), len(ptoks)
    else:
        ptoks = list(record["prompt_token_ids"])
        score_start = int(record.get("score_token_start",
                                     record.get("target_start", record.get("score_start", 1))))
        score_end = int(record.get("score_token_end",
                                   record.get("target_end", len(ptoks))))
    score_start = max(score_start, 1)
    assert ptoks and score_start < score_end <= len(ptoks), f"bad ppl record {rid}"
    return {"id": rid, "prompt_token_ids": ptoks,
            "score_start": score_start, "score_end": score_end}


def compute_ppl(llm, sampling_cls) -> dict:
    records = [normalized_ppl_record(json.loads(ln), i)
               for i, ln in enumerate(PPL_DATA.read_text().splitlines()) if ln.strip()]
    sp = sampling_cls(max_tokens=1, temperature=0.0, prompt_logprobs=1)
    prompts = [{"prompt_token_ids": r["prompt_token_ids"]} for r in records]
    outs = llm.generate(prompts, sp, use_tqdm=False)
    total_nll = 0.0
    total_tok = 0
    per_record = []
    for r, out in zip(records, outs):
        plps = out.prompt_logprobs  # list[None | {tok_id: Logprob}]
        ptoks = r["prompt_token_ids"]
        nll = 0.0
        ntok = 0
        for i in range(r["score_start"], r["score_end"]):
            entry = plps[i] if plps and i < len(plps) else None
            if entry is None or ptoks[i] not in entry:
                # actual token must be present (prompt_logprobs always includes it); guard anyway
                raise ValueError(f"ppl {r['id']}: missing logprob for pos {i} tok {ptoks[i]}")
            nll += -lp_value(entry[ptoks[i]])
            ntok += 1
        total_nll += nll
        total_tok += ntok
        per_record.append({"id": r["id"], "num_tokens": ntok, "nll": nll,
                           "ppl": math.exp(nll / ntok)})
    return {"ppl": math.exp(total_nll / total_tok),
            "mean_record_ppl": sum(p["ppl"] for p in per_record) / len(per_record),
            "num_records": len(records), "num_tokens": total_tok,
            "neg_log_likelihood": total_nll, "dataset": str(PPL_DATA)}


# ---- (1) tau=0.3 self-consistency gap_probe (#720 protocol) ----


def load_streams(arm_dir: Path) -> dict:
    rows = {}
    for ln in (arm_dir / "decode_outputs.jsonl").read_text().splitlines():
        if ln.strip():
            r = json.loads(ln)
            rows[r["id"]] = r
    return rows


def first_div(a: list[int], b: list[int]) -> int:
    for i in range(min(len(a), len(b))):
        if a[i] != b[i]:
            return i
    return min(len(a), len(b))


def gap_probe(llm, sampling_cls, ar_dir: Path, spec_dir: Path) -> dict:
    ar = load_streams(ar_dir)
    spec = load_streams(spec_dir)
    ids = sorted(set(ar) & set(spec))
    probes = []  # (id, dataset_index, d, ref_tok, cur_tok, shared_prefix_ids)
    for pid in ids:
        a = ar[pid]["completion_token_ids"]
        s = spec[pid]["completion_token_ids"]
        if ar[pid]["completion_token_sha256"] == spec[pid]["completion_token_sha256"]:
            continue
        d = first_div(a, s)
        if d >= len(a) or d >= len(s):
            # one is a strict prefix of the other (length divergence); skip the margin probe
            probes.append({"id": pid, "first_div": d, "ref_tok": None, "cur_tok": None,
                           "shared": ar[pid]["prompt_token_ids"] + a[:d], "len_div": True})
            continue
        probes.append({"id": pid, "first_div": d, "ref_tok": a[d], "cur_tok": s[d],
                       "dataset_index": spec[pid].get("dataset_index"),
                       "shared": ar[pid]["prompt_token_ids"] + a[:d], "len_div": False})

    sp = sampling_cls(max_tokens=1, temperature=0.0, logprobs=20)
    margin_probes = [p for p in probes if not p["len_div"]]
    prompts = [{"prompt_token_ids": p["shared"]} for p in margin_probes]
    outs = llm.generate(prompts, sp, use_tqdm=False) if prompts else []

    records = []
    confident = 0
    max_gap = 0.0
    gap_hist: dict[str, int] = {}
    for p, out in zip(margin_probes, outs):
        lps = out.outputs[0].logprobs[0]  # {tok_id: Logprob} top-20 at position d
        ranked = sorted(lps.items(), key=lambda kv: lp_value(kv[1]), reverse=True)
        top1_tok, top1_lp = ranked[0][0], lp_value(ranked[0][1])
        top2_tok, top2_lp = (ranked[1][0], lp_value(ranked[1][1])) if len(ranked) > 1 else (None, float("-inf"))
        gap = top1_lp - top2_lp
        is_conf = gap >= TAU
        confident += int(is_conf)
        max_gap = max(max_gap, gap)
        gk = f"{round(gap, 3)}"
        gap_hist[gk] = gap_hist.get(gk, 0) + 1
        ref_lp = lp_value(lps.get(p["ref_tok"])) if p["ref_tok"] in lps else None
        cur_lp = lp_value(lps.get(p["cur_tok"])) if p["cur_tok"] in lps else None
        ref_rank = next((i for i, (t, _) in enumerate(ranked) if t == p["ref_tok"]), None)
        cur_rank = next((i for i, (t, _) in enumerate(ranked) if t == p["cur_tok"]), None)
        pair_is_top2 = {p["ref_tok"], p["cur_tok"]} == {top1_tok, top2_tok}
        records.append({
            "id": p["id"], "dataset_index": p.get("dataset_index"), "first_div": p["first_div"],
            "ref_tok": p["ref_tok"], "cur_tok": p["cur_tok"],
            "top1_logprob": top1_lp, "top2_logprob": top2_lp, "gap_nat": gap,
            "confident": is_conf,
            "ref_rank": ref_rank, "cur_rank": cur_rank,
            "ref_logprob": ref_lp, "cur_logprob": cur_lp,
            "pair_is_model_top2": pair_is_top2,
        })
    n_len_div = sum(1 for p in probes if p["len_div"])
    return {
        "tau": TAU, "n_diverging": len(probes), "n_probed": len(margin_probes),
        "n_len_divergence_only": n_len_div,
        "confident_genuine_flips": confident,
        "max_gap_nat": max_gap,
        "gap_nat_histogram": dict(sorted(gap_hist.items(), key=lambda kv: float(kv[0]))),
        "frac_pair_is_model_top2": (sum(r["pair_is_model_top2"] for r in records) / len(records)) if records else None,
        "self_consistent_pass": confident == 0,
        "records": records,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bi", type=int, required=True, choices=(0, 1))
    ap.add_argument("--do-ppl", type=int, default=1, choices=(0, 1))
    ap.add_argument("--do-gapprobe", type=int, default=1, choices=(0, 1),
                    help="gap_probe is only meaningful for the BI=1 spec-vs-AR pair")
    ap.add_argument("--ar-dir", default="bi1_spec0")
    ap.add_argument("--spec-dir", default="bi1_spec1")
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    ap.add_argument("--max-num-batched-tokens", type=int, default=512,
                    help="chunk the prefill so the full-vocab (262k) prompt_logprobs logits "
                         "tensor stays small; un-chunked a 2943-tok prompt OOMs (1.34 GiB logits)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["VLLM_BATCH_INVARIANT"] = str(args.bi)
    os.environ["VLLM_ATTENTION_BACKEND"] = "TRITON_ATTN"
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    from vllm import LLM, SamplingParams

    t0 = time.time()
    llm = LLM(model=MODEL_ID, quantization="compressed-tensors", dtype="bfloat16",
              max_model_len=4096, max_num_seqs=1, gpu_memory_utilization=args.gpu_mem_util,
              enable_chunked_prefill=True, max_num_batched_tokens=args.max_num_batched_tokens,
              enforce_eager=True, trust_remote_code=True, seed=0)
    boot_s = time.time() - t0
    print(f"[selfconsist_ppl bi={args.bi}] model loaded in {boot_s:.0f}s", flush=True)

    result = {"bi": args.bi, "model_id": MODEL_ID, "enforce_eager": True,
              "attention_backend": "TRITON_ATTN", "boot_s": round(boot_s, 1)}

    if args.do_ppl:
        t = time.time()
        ppl = compute_ppl(llm, SamplingParams)
        ppl["wall_s"] = round(time.time() - t, 1)
        result["ppl"] = ppl
        print(f"[ppl bi={args.bi}] ppl={ppl['ppl']:.4f} "
              f"(mean_record={ppl['mean_record_ppl']:.4f}, tok={ppl['num_tokens']}, "
              f"{ppl['wall_s']}s)", flush=True)

    if args.do_gapprobe and args.bi == 1:
        t = time.time()
        runs = HERE / "runs"
        gp = gap_probe(llm, SamplingParams, runs / args.ar_dir, runs / args.spec_dir)
        gp["wall_s"] = round(time.time() - t, 1)
        result["gap_probe"] = gp
        print(f"[gap_probe bi=1] n_diverging={gp['n_diverging']} n_probed={gp['n_probed']} "
              f"confident_genuine_flips={gp['confident_genuine_flips']} "
              f"max_gap={gp['max_gap_nat']:.3f}nat self_consistent={gp['self_consistent_pass']} "
              f"({gp['wall_s']}s)", flush=True)
        print(f"  gap histogram (nat): {gp['gap_nat_histogram']}", flush=True)

    out = args.out or (HERE / "runs" / f"selfconsist_ppl_bi{args.bi}.json")
    out.write_text(json.dumps(result, indent=2))
    print(f"[selfconsist_ppl bi={args.bi}] -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
