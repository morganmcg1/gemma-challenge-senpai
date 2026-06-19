#!/usr/bin/env python
"""PR #729 -- fp8-KV orthogonal long-output decode lever, LOCAL A10G analysis.

ONE arm = one kv_cache_dtype. Loads the LOCKED int4_g128_lmhead checkpoint once
(same flags as serve.py: dtype bf16, gpu_mem_util 0.90, max_num_batched_tokens 512)
and runs three phases:

  A. TPS sweep at output_len in {512, 2048, 8192}, conc=1, temp=0, ignore_eos
     -> served output_tps per length (512 == official scoring point).
  B. PPL over the 128 ground-truth records (context+target, score target 512 toks)
     AND, from the SAME teacher-forced pass, per-position argmax + top1/top2 gap
     over the scored region (matched-state, cascade-free; #576 methodology).
  C. free-run greedy capture (sharegpt prompts, L=512) in check_greedy_identity
     format -> feeds the OFFICIAL byte-exact greedy-identity compare.

analysis_only=1, official_tps=0. No served file change, no HF Job.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

CKPT = "/workspace/gemma_build/int4_g128_lmhead"
SHAREGPT = ("official/main_bucket/shared_resources/speed_benchmark/data/"
            "eval_prompts_sharegpt.json")
PPL_DATA = ("official/main_bucket/shared_resources/speed_benchmark/data/"
            "ppl_ground_truth_tokens.jsonl")


def sha256_tokens(token_ids):
    return hashlib.sha256(",".join(str(t) for t in token_ids).encode("ascii")).hexdigest()


def to_ids(x):
    """Robustly flatten apply_chat_template output to list[int]."""
    if hasattr(x, "input_ids"):
        x = x.input_ids
    if isinstance(x, dict):
        x = x.get("input_ids", x)
    if hasattr(x, "tolist"):
        x = x.tolist()
    if isinstance(x, list) and x and isinstance(x[0], list):
        x = x[0]
    return [int(t) for t in x]


def chat_ids(tok, text):
    return to_ids(tok.apply_chat_template(
        [{"role": "user", "content": text}],
        add_generation_prompt=True, tokenize=True))


def read_sharegpt(path, n, seed):
    data = json.loads(Path(path).read_text())
    recs = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        conv = item.get("conversations")
        if not isinstance(conv, list) or len(conv) < 2 or not isinstance(conv[0], dict):
            continue
        p = conv[0].get("value")
        if isinstance(p, str) and p:
            recs.append({"id": str(item.get("id", index)), "index": index, "prompt_text": p})
    random.Random(seed).shuffle(recs)
    return recs[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kv-dtype", required=True)  # auto | fp8 | fp8_e5m2
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-model-len", type=int, default=9216)
    ap.add_argument("--tps-lengths", default="512,2048,8192")
    ap.add_argument("--tps-prompts", default="16,6,3")  # per length
    ap.add_argument("--greedy-prompts", type=int, default=32)
    ap.add_argument("--ppl-records", type=int, default=128)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--eager", action="store_true",
                    help="enforce_eager=True (bypass torch.compile/CUDA-graph) -- "
                         "used to prove the fp8e4nv KV failure is a HW/Triton dtype "
                         "limit, not a compile-autotune artifact")
    args = ap.parse_args()

    if args.debug:
        args.tps_lengths, args.tps_prompts = "8", "1"
        args.greedy_prompts, args.ppl_records = 2, 2
        args.max_model_len = 4096

    import torch
    from vllm import LLM, SamplingParams, TokensPrompt
    from transformers import AutoTokenizer

    t_load0 = time.time()
    llm = LLM(
        model=CKPT,
        dtype="bfloat16",
        kv_cache_dtype=args.kv_dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.90,
        max_num_batched_tokens=512,
        trust_remote_code=True,
        enforce_eager=args.eager,
        disable_log_stats=True,
    )
    load_s = time.time() - t_load0
    tok = AutoTokenizer.from_pretrained(CKPT)

    result = {
        "kv_dtype": args.kv_dtype, "ckpt": CKPT, "load_s": load_s,
        "max_model_len": args.max_model_len, "analysis_only": True, "official_tps": 0,
    }

    # ---------- Phase A: TPS sweep ----------
    Ls = [int(x) for x in args.tps_lengths.split(",")]
    Ns = [int(x) for x in args.tps_prompts.split(",")]
    sg = read_sharegpt(SHAREGPT, max(Ns), seed=7)
    tps = {}
    for L, N in zip(Ls, Ns):
        sp = SamplingParams(temperature=0.0, max_tokens=L, ignore_eos=True)
        per_req = []
        out_tok = 0
        for rec in sg[:N]:
            ids = chat_ids(tok, rec["prompt_text"])
            torch.cuda.synchronize()
            t0 = time.time()
            o = llm.generate([TokensPrompt(prompt_token_ids=ids)], sp, use_tqdm=False)
            torch.cuda.synchronize()
            dt = time.time() - t0
            n_out = len(o[0].outputs[0].token_ids)
            per_req.append({"prompt_len": len(ids), "n_out": n_out, "wall_s": dt})
            out_tok += n_out
        total_wall = sum(r["wall_s"] for r in per_req)
        tps[str(L)] = {
            "output_len": L, "n_prompts": N,
            "output_tps": out_tok / total_wall,
            "total_out_tokens": out_tok, "total_wall_s": total_wall,
            "per_req": per_req,
        }
        print(f"[A] L={L} N={N} output_tps={tps[str(L)]['output_tps']:.3f}", flush=True)
    result["tps"] = tps

    # ---------- Phase B: PPL + matched-state argmax/gap ----------
    ppl_lines = Path(PPL_DATA).read_text().strip().splitlines()[: args.ppl_records]
    recs = [json.loads(l) for l in ppl_lines]
    sp_ppl = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=20)
    prompts = []
    meta = []
    for i, r in enumerate(recs):
        ctx = [int(t) for t in r["context_token_ids"]]
        tgt = [int(t) for t in r["target_token_ids"]]
        ids = ctx + tgt
        prompts.append(TokensPrompt(prompt_token_ids=ids))
        meta.append({"id": str(r.get("id", i)), "ids": ids,
                     "score_start": len(ctx), "score_end": len(ids)})
    outs = llm.generate(prompts, sp_ppl, use_tqdm=False)

    total_nll = 0.0
    total_tok = 0
    per_record = []
    for m, o in zip(meta, outs):
        plp = o.prompt_logprobs  # list len == prompt len; [0] is None
        ids = m["ids"]
        ss, se = m["score_start"], m["score_end"]
        nll = 0.0
        argmax_ids = []
        gaps = []
        for pos in range(ss, se):
            entry = plp[pos]  # dict {token_id: Logprob}
            actual = ids[pos]
            # logprob of the actual token
            lp_actual = entry[actual].logprob
            nll += -lp_actual
            # top1/top2 by logprob -> argmax + confidence gap
            ranked = sorted(entry.values(), key=lambda x: x.logprob, reverse=True)
            top1 = ranked[0]
            top2 = ranked[1] if len(ranked) > 1 else ranked[0]
            # recover top1 token id
            top1_id = next(tid for tid, lg in entry.items() if lg is top1)
            argmax_ids.append(int(top1_id))
            gaps.append(float(top1.logprob - top2.logprob))
        total_nll += nll
        total_tok += (se - ss)
        per_record.append({
            "id": m["id"], "n_score": se - ss,
            "nll": nll, "ppl": math.exp(nll / (se - ss)),
            "argmax_ids": argmax_ids, "gaps": gaps,
        })
    result["ppl"] = {
        "ppl": math.exp(total_nll / total_tok),
        "num_records": len(recs), "num_tokens": total_tok,
        "neg_log_likelihood": total_nll,
    }
    result["matched_state"] = {"per_record": per_record}
    print(f"[B] PPL={result['ppl']['ppl']:.4f} over {total_tok} tokens", flush=True)

    # ---------- Phase C: free-run greedy capture (official gate format) ----------
    gsg = read_sharegpt(SHAREGPT, args.greedy_prompts, seed=1)  # seed=1 == official
    sp_g = SamplingParams(temperature=0.0, max_tokens=512, ignore_eos=True)
    g_prompts = []
    g_meta = []
    for index, rec in enumerate(gsg):
        ids = chat_ids(tok, rec["prompt_text"])
        g_prompts.append(TokensPrompt(prompt_token_ids=ids))
        g_meta.append({"id": rec["id"], "index": index})
    g_outs = llm.generate(g_prompts, sp_g, use_tqdm=False)
    greedy = []
    for m, o in zip(g_meta, g_outs):
        comp = [int(t) for t in o.outputs[0].token_ids]
        greedy.append({"id": m["id"], "index": m["index"],
                       "completion_token_ids": comp,
                       "completion_token_sha256": sha256_tokens(comp),
                       "num_completion_tokens": len(comp)})
    result["greedy_freerun"] = greedy
    print(f"[C] captured {len(greedy)} free-run greedy completions", flush=True)

    # peak memory
    result["peak_gib"] = torch.cuda.max_memory_allocated() / (1024 ** 3)
    result["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / (1024 ** 3)

    Path(args.out).write_text(json.dumps(result))
    print(f"WROTE {args.out}  peak_gib={result['peak_gib']:.2f}", flush=True)


if __name__ == "__main__":
    main()
