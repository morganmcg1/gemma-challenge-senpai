#!/usr/bin/env python3
"""PR #699 diagnostic W&B run: the int4 SAMPLED-basis 2x2 is BLOCKED by an
int4-specific decode degeneration on the substitute engine (vLLM 0.22.0).

This logs the evidence that settles WHY no sampled 2x2 surfaced:
  - matched base-vs-int4 probe (same 6 AIME-2024 problems, thinking=False, k=8,
    8192 tok): base decodes cleanly, int4 collapses into repetition/gibberish;
  - batch-width-1 isolation control: collapse persists at batch width 1, so it is
    NOT a batched-Marlin / batch-invariance artifact -- it is intrinsic to int4 x
    T=1.0 sampling on 0.22.0;
  - the pinned engine (0.22.1rc1.dev307, which produced lawine's clean 0.3467
    sampled anchor) is irrecoverable locally (evicted uv cache).

analysis_only: NO HF job, NO official TPS, fires=0.
"""
import json
from collections import Counter
from pathlib import Path

import wandb

HERE = Path(__file__).resolve().parent
SMOKE = HERE / "_smoke"


def rep_max(t, w=40):
    if len(t) < 200:
        return 0.0
    c = Counter(t[i : i + w] for i in range(0, len(t) - w, w))
    return c.most_common(1)[0][1] * w / len(t)


def summarize(path):
    d = json.load(open(path))
    pp = d["per_problem"]
    fr = [r for p in pp for r in p["finish_reasons"]]
    n = len(fr)
    trunc = fr.count("length")
    # a "length" sample is degenerate if its body is a repetition loop
    gib = tot = 0
    for p in pp:
        for t, f in zip(p["texts"], p["finish_reasons"]):
            if f == "length":
                tot += 1
                if rep_max(t) > 0.30:
                    gib += 1
    return {
        "maj_k_accuracy": d["maj_k_accuracy"],
        "mean_pass_rate": d["mean_pass_rate"],
        "extract_fail_rate": d["extract_fail_rate"],
        "trunc_rate": trunc / n,
        "n_samples": n,
        "gibberish_among_trunc": (gib / tot) if tot else 0.0,
        "k": d["k"],
        "max_tokens": d["sampling"]["max_tokens"],
    }


base = summarize(SMOKE / "base_smoke.json")
int4 = summarize(SMOKE / "int4_match.json")

# batch-width-1 control: count length-truncated samples (isolated, k=1, cc=1)
bw1 = json.load(open(SMOKE / "int4_bw1_s1234.json"))
bw1_n = sum(len(p["finish_reasons"]) for p in bw1["per_problem"])
bw1_len = sum(p["finish_reasons"].count("length") for p in bw1["per_problem"])

trunc_delta = int4["trunc_rate"] - base["trunc_rate"]

config = {
    "analysis_only": 1,
    "official_tps": 0,
    "no_hf_job": 1,
    "fires": 0,
    # decode provenance (so this reconciles against lawine's 0.3467 anchor)
    "eval_decode_basis": "sampled_lewtun31",
    "eval_sampling": "T1.0_top_p0.95_top_k64",
    "eval_temperature": 1.0,
    "eval_top_p": 0.95,
    "eval_top_k": 64,
    "eval_min_tokens": 8,
    "eval_max_tokens_probe": int4["max_tokens"],
    # engine
    "engine_used": "vllm-0.22.0",
    "engine_pinned": "vllm-0.22.1rc1.dev307+g3e8afdf78.cu129",
    "engine_pinned_status": "DEAD_evicted_uv_cache_341_dangling_symlinks",
    "batch_invariant": 0,
    "flashinfer_sampler": 0,
    "int4_build": "/workspace/gemma_build/int4_g128_lmhead",
    "subset": "aime2024_6problems_matched",
    # relayed anchors (advisor relay #693; NOT fetched)
    "anchor_lawine_int4_sampled_12288": 0.3467,
    "anchor_lawine_base_sampled_12288": 0.4600,
    # banked greedy reproduction (kanna comment 01:26Z, n=60)
    "greedy_banked_int4_6144": 0.350,
    "greedy_banked_base_6144": 0.4667,
    "greedy_banked_trunc_delta_6144": 0.034,
}

run = wandb.init(
    project="gemma-challenge-senpai",
    entity="wandb-applied-ai-team",
    group="int4-aime-budget-artifact-kanna",
    name="kanna/int4-aime-budget-artifact-decode-blocker",
    job_type="diagnostic",
    config=config,
    tags=["pr699", "kanna", "analysis_only", "decode-blocker", "int4-sampled-degeneration"],
)

wandb.log(
    {
        # matched base-vs-int4 (the smoking gun)
        "base_maj8": base["maj_k_accuracy"],
        "base_pass_rate": base["mean_pass_rate"],
        "base_extract_fail": base["extract_fail_rate"],
        "base_trunc_rate": base["trunc_rate"],
        "base_gibberish_among_trunc": base["gibberish_among_trunc"],
        "int4_maj8": int4["maj_k_accuracy"],
        "int4_pass_rate": int4["mean_pass_rate"],
        "int4_extract_fail": int4["extract_fail_rate"],
        "int4_trunc_rate": int4["trunc_rate"],
        "int4_gibberish_among_trunc": int4["gibberish_among_trunc"],
        # REQUIRED test-metric proxy (matched 6-subset, sampled)
        "aime_truncation_rate_delta_int4_vs_base": trunc_delta,
        # batch-width-1 isolation control
        "bw1_trunc_rate": bw1_len / bw1_n,
        "bw1_extract_fail": bw1["extract_fail_rate"],
        # verdict flags
        "sampled_2x2_status_blocked": 1,
        "degeneration_int4_specific": 1,
        "degeneration_batchwidth_dependent": 0,
        "reconciles_to_0p3467_anchor": 0,
        # primary metric cannot be computed on this engine
        "aime_int4_pct_of_base_at_12288": float("nan"),
    }
)

print("WANDB_RUN_ID", run.id)
print("WANDB_RUN_URL", run.url)
print(f"trunc_delta(matched 6-subset) = {trunc_delta:+.3f}")
print(f"base   maj8={base['maj_k_accuracy']:.3f} xfail={base['extract_fail_rate']:.3f} trunc={base['trunc_rate']:.3f} gib_among_trunc={base['gibberish_among_trunc']:.3f}")
print(f"int4   maj8={int4['maj_k_accuracy']:.3f} xfail={int4['extract_fail_rate']:.3f} trunc={int4['trunc_rate']:.3f} gib_among_trunc={int4['gibberish_among_trunc']:.3f}")
print(f"bw1    trunc={bw1_len}/{bw1_n} xfail={bw1['extract_fail_rate']:.3f}")
wandb.finish()
