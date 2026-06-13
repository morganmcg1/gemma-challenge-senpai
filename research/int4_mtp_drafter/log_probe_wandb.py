#!/usr/bin/env python
"""Probe a running spec-decode endpoint and log TPS + acceptance to W&B.

Single-stream (concurrency=1), greedy (temp=0), integer-token prompt. Times
output throughput and reads vLLM /metrics spec-decode counters to derive mean
accepted tokens/step and the per-position acceptance curve, then logs everything
to one W&B run (one run per num_speculative_tokens K, shared group).

Local AWS A10G exploratory numbers only -- NOT the official a10g-small score.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request


def get(url: str, timeout: float = 10.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def post(url: str, payload: dict, timeout: float = 600.0) -> dict:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def scrape_spec_metrics(base: str) -> dict:
    out: dict[str, float] = {}
    per_pos: dict[int, float] = {}
    try:
        text = get(f"{base}/metrics", timeout=10)
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        # Skip Prometheus auxiliary series: ``_created`` carries the metric's
        # creation unix-timestamp (~1.78e9), ``_sum``/``_count``/``_bucket`` are
        # histogram internals. Only the raw counter (bare name or ``_total``) is a
        # real count. Without this, the per-position scrape below summed the
        # ~1.78e9 ``_created`` timestamp into each position's accept count; the
        # aggregate counters happened to survive because ``d()`` differences
        # before/after and the constant timestamp cancels, but the un-differenced
        # per_pos curve was pure garbage.
        metric_name = line.split("{", 1)[0].split(" ", 1)[0]
        if metric_name.endswith(("_created", "_sum", "_count", "_bucket")):
            continue
        for key in (
            "vllm:spec_decode_num_draft_tokens",
            "vllm:spec_decode_num_accepted_tokens",
            "vllm:spec_decode_num_drafts",
        ):
            if line.startswith(key) and "_per_pos" not in line:
                try:
                    out[key] = out.get(key, 0.0) + float(line.rsplit(" ", 1)[1])
                except ValueError:
                    pass
        if "spec_decode_num_accepted_tokens_per_pos" in line:
            try:
                pos = int(line.split('position="', 1)[1].split('"', 1)[0])
                per_pos[pos] = per_pos.get(pos, 0.0) + float(line.rsplit(" ", 1)[1])
            except (ValueError, IndexError):
                pass
    if per_pos:
        out["per_pos"] = [per_pos[k] for k in sorted(per_pos)]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="gemma-4-e4b-it")
    ap.add_argument("--k", type=int, required=True, help="num_speculative_tokens")
    ap.add_argument("--group", default="int4-mtp-drafter")
    ap.add_argument("--name", default=None)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--prompt", default="2,2364,1841,603,573,2669,576,3777,235336")
    ap.add_argument("--greedy-identical", default=None, help="true/false/unknown")
    ap.add_argument("--ppl", type=float, default=None)
    args = ap.parse_args()

    prompt_ids = [int(x) for x in args.prompt.split(",") if x.strip()]
    base = args.base.rstrip("/")

    before = scrape_spec_metrics(base)
    times, ntok = [], []
    for i in range(args.reps):
        payload = {
            "model": args.model,
            "prompt": prompt_ids,
            "max_tokens": args.max_tokens,
            "temperature": 0.0,
            "stream": False,
            "add_special_tokens": False,
            "ignore_eos": True,
            "return_token_ids": True,
        }
        t0 = time.time()
        resp = post(f"{base}/v1/completions", payload)
        dt = time.time() - t0
        ch = resp["choices"][0]
        toks = ch.get("token_ids") or ch.get("output_token_ids") or []
        gen = len(toks) - (len(prompt_ids) if toks[: len(prompt_ids)] == prompt_ids else 0) if toks else args.max_tokens
        times.append(dt)
        ntok.append(gen)
        print(f"rep{i}: {gen} tok in {dt:.2f}s -> {gen / dt:.2f} tok/s")

    after = scrape_spec_metrics(base)

    def d(k: str) -> float:
        return float(after.get(k, 0.0)) - float(before.get(k, 0.0))

    draft = d("vllm:spec_decode_num_draft_tokens")
    acc = d("vllm:spec_decode_num_accepted_tokens")
    drafts = d("vllm:spec_decode_num_drafts")
    best_dt = min(times)
    best_tok = ntok[times.index(best_dt)]
    best_tps = best_tok / best_dt
    mean_acc_per_step = (1.0 + acc / drafts) if drafts > 0 else None
    overall_accept = (acc / draft) if draft > 0 else None
    # Difference per-position accept counts before/after the probe (isolates the
    # probe's reps from any startup/profiling generations), then normalize by the
    # number of draft steps to get the cumulative acceptance-at-depth curve.
    bpp, app = before.get("per_pos"), after.get("per_pos")
    if app and bpp and len(app) == len(bpp):
        per_pos = [a - b for a, b in zip(app, bpp)]
    else:
        per_pos = app
    per_pos_rate = [p / drafts for p in per_pos] if (per_pos and drafts > 0) else None

    print("\n=== SUMMARY ===")
    print(f"best single-stream TPS (local A10G, exploratory): {best_tps:.2f} tok/s")
    print(f"spec delta: drafts={drafts} draft_tokens={draft} accepted_tokens={acc}")
    if mean_acc_per_step is not None:
        print(f"mean accepted tokens / step (incl. bonus): {mean_acc_per_step:.3f}")
    if overall_accept is not None:
        print(f"overall acceptance rate: {overall_accept:.3f}")
    print(f"per_pos accepted (cumulative counters): {per_pos}")
    print(f"per_pos acceptance rate: {per_pos_rate}")

    try:
        import wandb
    except ModuleNotFoundError:
        print("wandb not installed; skipping W&B logging")
        return

    gi = args.greedy_identical
    gi_bool = {"true": True, "false": False}.get((gi or "").lower(), None)
    name = args.name or f"kanna/int4-mtp-drafter-k{args.k}"
    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "senpai-v1"),
        entity=os.environ.get("WANDB_ENTITY") or None,
        group=args.group,
        name=name,
        job_type="specdecode-probe",
        config={
            "num_speculative_tokens": args.k,
            "model_id": "google/gemma-4-E4B-it-qat-w4a16-ct",
            "drafter": "google/gemma-4-E4B-it-qat-q4_0-unquantized-assistant",
            "engine": "vllm==0.22.0",
            "max_model_len": 4096,
            "max_num_batched_tokens": 512,
            "max_num_seqs": 1,
            "gpu": "A10G (local exploratory)",
            "output_len": args.max_tokens,
            "reps": args.reps,
            "spec_method": "mtp",
        },
    )
    log = {
        "exploratory_tps_a10g": best_tps,
        "spec/drafts": drafts,
        "spec/draft_tokens": draft,
        "spec/accepted_tokens": acc,
    }
    if mean_acc_per_step is not None:
        log["spec/mean_accepted_tokens_per_step"] = mean_acc_per_step
    if overall_accept is not None:
        log["spec/overall_acceptance_rate"] = overall_accept
    if gi_bool is not None:
        log["greedy_identical"] = int(gi_bool)
    if args.ppl is not None:
        log["ppl"] = args.ppl
    if per_pos_rate:
        for i, r in enumerate(per_pos_rate):
            log[f"spec/accept_rate_pos{i}"] = r
    wandb.log(log)
    run.summary.update(log)
    print(f"logged to W&B run: {run.url}")
    run.finish()


if __name__ == "__main__":
    main()
