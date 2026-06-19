#!/usr/bin/env python3
"""PR #700 instr #1: per-module ACTIVATION-norm probe on an AIME calibration set.

Runs a small AIME calibration set through the int4 body (g32 QAT-w4a16-ct, the
on-disk int4 reference; activations are int4-faithful and the per-module RANKING
is invariant to g32-vs-g128, both being int4 of the SAME QAT source to rel<0.07).
For each of the 343 body Linear modules captures the mean L2 norm of its INPUT
activation, averaged over tokens x calibration prompts -- the activation-
reweighting signal `act_norm` for impact = rel_div x act_norm (#700 instr #2).

Two regimes captured for a validity check that prompt-position norms rank like
the reasoning-DECODE regime (AIME reasoning happens during decode, not prefill):
  * PREFILL  -- one forward over each chat-templated prompt (all 30 prompts).
  * DECODE   -- short greedy decode (a subset of prompts) -> decode-position norms.
Reports Spearman(prefill_rank, decode_rank); high rho validates prefill as the
cheap primary measurement.

analysis_only: NO HF Job, NO submission, served file untouched, on-disk ckpt only.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path

import torch

G32_DIR = (
    "/senpai-run/home/student-ubel/.cache/huggingface/hub/"
    "models--google--gemma-4-E4B-it-qat-w4a16-ct/snapshots/"
    "ef0a4c43726bde42a3ca04fd300397c0b8b3c3f0"
)
REF695 = Path(__file__).parent / "ref695" / "sqnr_probe.json"

AIME_INSTRUCTION = (
    "Please reason step by step to solve the problem, and put your final answer "
    "(a single integer between 0 and 999) within \\boxed{}."
)
# Calibration set: AIME-2024 (Maxwell-Jia/AIME_2024, the same 30-problem set the
# base AIME measurement used). The activation-norm per-module RANKING is robust to
# the exact problem choice; 30 items gives full-module-set poolability (#700 #1).
AIME_DATASET = "Maxwell-Jia/AIME_2024"
AIME_CONFIG = "default"
AIME_SPLIT = "train"
AIME_PROBLEM_COL = "Problem"


def load_aime(n: int) -> list[str]:
    url = (
        "https://datasets-server.huggingface.co/rows"
        f"?dataset={urllib.parse.quote(AIME_DATASET)}&config={AIME_CONFIG}"
        f"&split={AIME_SPLIT}&offset=0&length=100"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "senpai-aime-actprobe"})
    tok = os.environ.get("HF_TOKEN")
    if tok:
        req.add_header("Authorization", f"Bearer {tok}")
    last = None
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.load(r)
            rows = [row["row"][AIME_PROBLEM_COL] for row in data.get("rows", [])]
            return [str(p) for p in rows[:n]]
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"AIME fetch failed: {last}")


BUCKETS = ("prefill", "decode", "decode_warm")


class NormAccumulator:
    """Per-module running sum of per-token input L2 norms + token count, per bucket.

    Routing: in the 'prefill' phase every call -> 'prefill'. In the 'decode' phase
    a call with exactly ONE token is an autoregressive decode step -> 'decode';
    a multi-token call is generate()'s internal prefill -> 'decode_warm' (kept out
    of the decode-position statistic)."""

    def __init__(self, names: list[str]):
        self.sum = {n: {b: 0.0 for b in BUCKETS} for n in names}
        self.cnt = {n: {b: 0 for b in BUCKETS} for n in names}
        self.in_dim = {n: 0 for n in names}
        self.phase = "prefill"

    def make_hook(self, name: str):
        def hook(mod, args):
            x = args[0]
            if x.dim() == 3:
                x = x.reshape(-1, x.shape[-1])
            norms = x.float().norm(dim=-1)  # per-token L2
            ntok = int(norms.numel())
            if self.phase == "prefill":
                bucket = "prefill"
            else:
                bucket = "decode" if ntok == 1 else "decode_warm"
            self.sum[name][bucket] += float(norms.sum())
            self.cnt[name][bucket] += ntok
            if self.in_dim[name] == 0:
                self.in_dim[name] = int(x.shape[-1])
        return hook


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-prompts", type=int, default=30, help="AIME calibration prompts (prefill)")
    ap.add_argument("--decode-subset", type=int, default=8, help="prompts for decode-regime validation")
    ap.add_argument("--decode-tokens", type=int, default=128, help="greedy new tokens per decode-subset prompt")
    ap.add_argument("--out", default=str(Path(__file__).parent / "act_norms.json"))
    args = ap.parse_args()

    t0 = time.time()
    ref = json.load(open(REF695))
    ref_mods = [r["module"] for r in ref["rows"]]
    print(f"[act] #695 ref modules: {len(ref_mods)}", flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(G32_DIR)
    model = AutoModelForCausalLM.from_pretrained(G32_DIR, dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()
    torch.cuda.synchronize()
    print(f"[act] model loaded, VRAM {torch.cuda.memory_allocated(0)/2**30:.2f} GiB ({time.time()-t0:.1f}s)", flush=True)

    named = dict(model.named_modules())
    missing = [m for m in ref_mods if m not in named]
    assert not missing, f"{len(missing)} ref modules missing in runtime, e.g. {missing[:3]}"

    acc = NormAccumulator(ref_mods)
    handles = [named[m].register_forward_pre_hook(acc.make_hook(m)) for m in ref_mods]
    print(f"[act] hooked {len(handles)} modules", flush=True)

    prompts = load_aime(args.n_prompts)
    print(f"[act] AIME calibration prompts: {len(prompts)}", flush=True)

    def encode(problem: str):
        messages = [{"role": "user", "content": f"{problem}\n\n{AIME_INSTRUCTION}"}]
        try:
            enc = tok.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt",
                return_dict=True, enable_thinking=True,
            )
        except TypeError:
            enc = tok.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt", return_dict=True,
            )
        ids = enc["input_ids"].to("cuda:0")
        am = enc.get("attention_mask")
        am = am.to("cuda:0") if am is not None else torch.ones_like(ids)
        return ids, am

    # ---- PREFILL phase: one forward per prompt, all prompts ----
    acc.phase = "prefill"
    prompt_lens = []
    with torch.no_grad():
        for i, p in enumerate(prompts):
            ids, am = encode(p)
            prompt_lens.append(int(ids.shape[1]))
            _ = model(ids, attention_mask=am)
            if (i + 1) % 10 == 0:
                print(f"[act] prefill {i+1}/{len(prompts)} (len {ids.shape[1]})", flush=True)
    print(f"[act] prefill done ({time.time()-t0:.1f}s); prompt_len mean {sum(prompt_lens)/len(prompt_lens):.0f}", flush=True)

    # ---- DECODE phase (validation subset): greedy decode -> single-token steps
    #      route to 'decode'; generate()'s internal prefill -> 'decode_warm'. ----
    acc.phase = "decode"
    dec_n = min(args.decode_subset, len(prompts))
    with torch.no_grad():
        for i in range(dec_n):
            ids, am = encode(prompts[i])
            _ = model.generate(
                ids, attention_mask=am, max_new_tokens=args.decode_tokens,
                do_sample=False, use_cache=True, pad_token_id=tok.eos_token_id,
            )
            if (i + 1) % 4 == 0:
                print(f"[act] decode {i+1}/{dec_n}", flush=True)
    print(f"[act] decode done ({time.time()-t0:.1f}s)", flush=True)

    for h in handles:
        h.remove()

    rows = []
    for m in ref_mods:
        pc, dc = acc.cnt[m]["prefill"], acc.cnt[m]["decode"]
        pn = acc.sum[m]["prefill"] / pc if pc else 0.0
        dn = acc.sum[m]["decode"] / dc if dc else 0.0
        rows.append({
            "module": m,
            "in_dim": acc.in_dim[m],
            "prefill_tokens": pc,
            "decode_tokens": dc,
            "act_norm_prefill": pn,
            "act_norm_decode": dn,
        })

    # Spearman rank correlation prefill vs decode (validates prefill as primary).
    def spearman(a, b):
        import statistics
        n = len(a)
        ra = {v: i for i, v in enumerate(sorted(range(n), key=lambda k: a[k]))}
        rb = {v: i for i, v in enumerate(sorted(range(n), key=lambda k: b[k]))}
        x = [ra[i] for i in range(n)]
        y = [rb[i] for i in range(n)]
        mx, my = statistics.mean(x), statistics.mean(y)
        num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
        den = (sum((xi - mx) ** 2 for xi in x) * sum((yi - my) ** 2 for yi in y)) ** 0.5
        return num / den if den else float("nan")

    pre = [r["act_norm_prefill"] for r in rows]
    dec = [r["act_norm_decode"] for r in rows]
    rho = spearman(pre, dec)

    out = {
        "meta": {
            "checkpoint": "gemma-4-E4B-it-qat-w4a16-ct (g32, int4 ref)",
            "n_prompts": len(prompts),
            "decode_subset": dec_n,
            "decode_tokens": args.decode_tokens,
            "prompt_len_mean": sum(prompt_lens) / len(prompt_lens),
            "aime_dataset": AIME_DATASET,
            "spearman_prefill_decode": rho,
            "elapsed_s": time.time() - t0,
        },
        "rows": rows,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[act] Spearman(prefill,decode) rank rho = {rho:.4f}", flush=True)
    print(f"[act] wrote {args.out} ({time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
