#!/usr/bin/env python
"""Offline ablation harness for the fa2sw + onegraph target-side runtime levers.

denken / PR #7 (branch denken/fa2sw-onegraph). One *variant* per process (the
levers are process-global: a config monkeypatch and a CUDA-graph capture mode
baked at engine construction). For --variant in {base, fa2sw, onegraph, both} it:

  1. applies the levers,
  2. records the per-layer attention backend map (head_size -> backend),
  3. greedy-decodes a fixed prompt set -> decode_outputs.jsonl (greedy-identity),
  4. measures single-stream decode TPS (warmup + repeats),
  5. computes teacher-forced PPL on the ground-truth tokens,

and writes summary.json. Compare variants' decode_outputs.jsonl against the base
variant with the official check_greedy_identity.py to get the GREEDY_IDENTICAL
verdict.

Levers
------
fa2sw  : neutralise Gemma4Config.verify_and_update_config, whose only job is to
         force TRITON_ATTN model-wide when head dims are heterogeneous. With it
         neutralised, per-head_size selection runs: the 35 sliding hd=256 layers
         pick FLASH_ATTN (FA2, which honours per_layer_sliding_window=512) and
         the 7 global hd=512 layers fall through to TRITON_ATTN (FA caps at 256).
onegraph: force cudagraph_mode=FULL (capture the whole decode step as one graph)
         instead of the FULL_AND_PIECEWISE default.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import time
from pathlib import Path

# The engine must stay in-process so the fa2sw monkeypatch applies to the worker,
# and the native sampler avoids a flashinfer JIT that needs a curand.h absent from
# this box's CUDA toolkit. Greedy decode is argmax, so the sampler backend cannot
# change tokens. Set before importing vllm.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

REPO = Path(__file__).resolve().parents[2]
PPL_DATA = (
    REPO
    / "official/main_bucket/shared_resources/speed_benchmark/data"
    / "ppl_ground_truth_tokens.jsonl"
)
MODEL_ID = os.environ.get("MODEL_ID", "google/gemma-4-E4B-it-qat-w4a16-ct")
TPS_PROMPT = (
    "Explain, step by step, how a transformer language model generates text "
    "one token at a time, and why decode is memory-bandwidth bound."
)


def sha256_tokens(token_ids: list[int]) -> str:
    """Authoritative harness recipe (see greedy_identity.sha256_tokens)."""
    return hashlib.sha256(
        ",".join(str(t) for t in token_ids).encode("ascii")
    ).hexdigest()


def install_backend_recorder() -> list[dict]:
    """Wrap Attention.__init__ to record each layer's (head_size, backend)."""
    import vllm.model_executor.layers.attention.attention as am

    records: list[dict] = []
    orig = am.Attention.__init__

    def patched(self, *args, **kwargs):
        orig(self, *args, **kwargs)
        try:
            records.append(
                {
                    "layer": getattr(self, "layer_name", ""),
                    "head_size": getattr(self, "head_size", None),
                    "backend": self.attn_backend.get_name(),
                }
            )
        except Exception:
            pass

    am.Attention.__init__ = patched
    return records


def apply_fa2sw() -> None:
    """Route the sliding hd=256 layers to FLASH_ATTN while keeping the global
    hd=512 layers on TRITON_ATTN. Two changes are needed:

    1. Neutralise Gemma4Config's heterogeneous-head-dim TRITON force-pin, so the
       backend stays None and per-head_size selection runs.
    2. Drop FLASHINFER from the sm_86 priority. Without it, the hd=512 global
       layers (FLASH_ATTN caps at 256) would pick FLASHINFER, whose kernel can't
       dispatch head_dim=512 (`Unsupported max_mma_kv: 0`) and crashes at the
       dummy run. With FLASHINFER gone they fall through to TRITON_ATTN.
    """
    import vllm.model_executor.models.config as cfg
    import vllm.platforms.cuda as cuda_mod

    def noop(vllm_config):  # noqa: ANN001 - matches staticmethod signature
        return None

    cfg.Gemma4Config.verify_and_update_config = staticmethod(noop)

    orig_priorities = cuda_mod._get_backend_priorities

    def no_flashinfer(*args, **kwargs):
        return [b for b in orig_priorities(*args, **kwargs) if b.name != "FLASHINFER"]

    cuda_mod._get_backend_priorities = no_flashinfer


def build_llm(variant: str, enable_prefix_caching: bool = True,
              enforce_eager: bool = False):
    from vllm import LLM

    kwargs = dict(
        model=MODEL_ID,
        quantization="compressed-tensors",
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.90,
        max_num_batched_tokens=512,
        max_num_seqs=1,
        enforce_eager=enforce_eager,
        trust_remote_code=True,
        disable_log_stats=True,
        enable_prefix_caching=enable_prefix_caching,
    )
    if variant in ("onegraph", "both") and not enforce_eager:
        # Force the whole decode step into a single full CUDA graph.
        # (No-op under enforce_eager, which disables all graph capture.)
        kwargs["compilation_config"] = {"cudagraph_mode": "FULL"}
    return LLM(**kwargs)


def load_ppl_records(limit: int | None) -> list[dict]:
    recs = [json.loads(l) for l in PPL_DATA.read_text().splitlines() if l.strip()]
    return recs if limit is None else recs[:limit]


def load_decode_prompts(path: str, limit: int | None) -> list[dict]:
    """Load greedy-identity prompts from a harness decode_outputs.jsonl, reusing
    its exact ``prompt_token_ids`` + ``id`` so the offline decode lands on the
    official audit prompt set (verifier keys match the served reference)."""
    recs = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        recs.append({"id": r["id"], "context_token_ids": r["prompt_token_ids"]})
    return recs if limit is None else recs[:limit]


def _greedy_decode(llm, records: list[dict], gen_tokens: int, sequential: bool) -> list[list[int]]:
    from vllm import SamplingParams

    sp = SamplingParams(temperature=0.0, max_tokens=gen_tokens, ignore_eos=True)
    prompts = [{"prompt_token_ids": r["context_token_ids"]} for r in records]
    if sequential:
        # One request at a time = the conc=1 serving condition (no cross-prompt
        # batching). Match plain greedy AR decode.
        outs = [llm.generate([p], sp)[0] for p in prompts]
    else:
        outs = llm.generate(prompts, sp)
    return [list(o.outputs[0].token_ids) for o in outs]


def run_greedy_identity(llm, records: list[dict], gen_tokens: int,
                        sequential: bool = False) -> list[dict]:
    """Greedy-decode gen_tokens from each record's context tokens."""
    all_ids = _greedy_decode(llm, records, gen_tokens, sequential)
    return [
        {
            "id": r["id"],
            "completion_token_ids": ids,
            "completion_token_sha256": sha256_tokens(ids),
        }
        for r, ids in zip(records, all_ids)
    ]


def determinism_check(llm, records: list[dict], gen_tokens: int, sequential: bool) -> dict:
    """Decode the same prompts twice in-process and compare token-for-token.
    A clean engine should be run-to-run deterministic for greedy decode; if not,
    cross-variant identity comparisons are confounded by kernel non-determinism."""
    a = _greedy_decode(llm, records, gen_tokens, sequential)
    b = _greedy_decode(llm, records, gen_tokens, sequential)
    diverged = sum(1 for x, y in zip(a, b) if x != y)
    tok_diff = sum(
        1 for x, y in zip(a, b) for i in range(min(len(x), len(y))) if x[i] != y[i]
    )
    return {
        "prompts": len(a),
        "divergent_prompts": diverged,
        "divergent_tokens": tok_diff,
        "deterministic": diverged == 0,
    }


def measure_tps(llm, tps_tokens: int, repeats: int) -> dict:
    import torch
    from vllm import SamplingParams

    sp = SamplingParams(temperature=0.0, max_tokens=tps_tokens, ignore_eos=True)
    # warmup
    llm.generate([TPS_PROMPT], SamplingParams(temperature=0.0, max_tokens=16, ignore_eos=True))
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        t = time.time()
        out = llm.generate([TPS_PROMPT], sp)
        torch.cuda.synchronize()
        wall = time.time() - t
        n = len(out[0].outputs[0].token_ids)
        samples.append(n / wall)
    mean = sum(samples) / len(samples)
    std = (sum((s - mean) ** 2 for s in samples) / len(samples)) ** 0.5
    return {"tps_mean": mean, "tps_std": std, "tps_samples": samples, "tps_tokens": tps_tokens}


def compute_ppl(llm, records: list[dict]) -> dict:
    """Teacher-forced PPL replicating ppl_endpoint.py exactly, but offline.

    For each record prompt = context + target; score positions [len(context),
    len(prompt)). NLL = -sum logprob(actual token); PPL = exp(sum_nll/sum_tok).
    """
    from vllm import SamplingParams

    sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1)
    prompts, metas = [], []
    for r in records:
        ctx, tgt = r["context_token_ids"], r["target_token_ids"]
        ids = ctx + tgt
        prompts.append({"prompt_token_ids": ids})
        metas.append((r["id"], ids, max(len(ctx), 1), len(ids)))
    outs = llm.generate(prompts, sp)

    total_nll = 0.0
    total_tok = 0
    per_record = []
    for (rid, ids, start, end), o in zip(metas, outs):
        plp = o.prompt_logprobs
        nll = 0.0
        ntok = 0
        for i in range(start, end):
            entry = plp[i] if i < len(plp) else None
            if not entry:
                raise ValueError(f"record {rid}: missing prompt_logprob at {i}")
            tok = ids[i]
            lp_obj = entry.get(tok)
            if lp_obj is None:
                raise ValueError(f"record {rid}: token {tok} absent at position {i}")
            lp = getattr(lp_obj, "logprob", lp_obj)
            nll += -float(lp)
            ntok += 1
        total_nll += nll
        total_tok += ntok
        per_record.append({"id": rid, "ppl": math.exp(nll / ntok), "num_tokens": ntok})
    return {
        "ppl": math.exp(total_nll / total_tok),
        "mean_record_ppl": sum(p["ppl"] for p in per_record) / len(per_record),
        "num_records": len(per_record),
        "num_tokens": total_tok,
        "neg_log_likelihood": total_nll,
        "per_record": per_record,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["base", "fa2sw", "onegraph", "both"])
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--n-identity", type=int, default=32, help="prompts for greedy-identity")
    ap.add_argument("--decode-prompts", default=None,
                    help="harness decode_outputs.jsonl to source identity prompts from "
                         "(uses its prompt_token_ids + id; default = aime/gpqa ppl set)")
    ap.add_argument("--gen-tokens", type=int, default=96, help="greedy tokens per identity prompt")
    ap.add_argument("--tps-tokens", type=int, default=256)
    ap.add_argument("--tps-repeats", type=int, default=3)
    ap.add_argument("--n-ppl", type=int, default=128, help="records for PPL (<=128)")
    ap.add_argument("--skip-ppl", action="store_true")
    ap.add_argument("--no-prefix-cache", action="store_true",
                    help="disable prefix caching (match plain AR decode, no cross-prompt reuse)")
    ap.add_argument("--sequential", action="store_true",
                    help="decode one request at a time (conc=1 serving condition)")
    ap.add_argument("--enforce-eager", action="store_true",
                    help="disable CUDA graphs + torch.compile (plain eager AR path; "
                         "the canonical 'plain greedy AR' reference)")
    ap.add_argument("--determinism-k", type=int, default=0,
                    help="if >0, double-decode the first K prompts in-process and report determinism")
    args = ap.parse_args()

    outdir = Path(args.outdir or (Path(__file__).resolve().parent / "runs" / args.variant))
    outdir.mkdir(parents=True, exist_ok=True)

    backend_records = install_backend_recorder()
    if args.variant in ("fa2sw", "both"):
        apply_fa2sw()

    t0 = time.time()
    llm = build_llm(args.variant, enable_prefix_caching=not args.no_prefix_cache,
                    enforce_eager=args.enforce_eager)
    load_s = time.time() - t0
    backend_summary = {f"{k[0]}|{k[1]}": v for k, v in collections.Counter(
        (r["head_size"], r["backend"]) for r in backend_records
    ).items()}
    (outdir / "backend_map.json").write_text(
        json.dumps({"summary": backend_summary, "layers": backend_records}, indent=2)
    )
    print(f"[{args.variant}] LOAD_OK {load_s:.1f}s  BACKEND={backend_summary}  "
          f"prefix_cache={not args.no_prefix_cache} sequential={args.sequential}", flush=True)

    # in-process determinism self-check
    det = None
    if args.determinism_k > 0:
        det = determinism_check(llm, load_ppl_records(args.determinism_k),
                                args.gen_tokens, args.sequential)
        print(f"[{args.variant}] DETERMINISM {det}", flush=True)

    # greedy-identity
    if args.decode_prompts:
        id_recs = load_decode_prompts(args.decode_prompts, args.n_identity)
    else:
        id_recs = load_ppl_records(args.n_identity)
    decode_rows = run_greedy_identity(llm, id_recs, args.gen_tokens, sequential=args.sequential)
    with (outdir / "decode_outputs.jsonl").open("w") as fh:
        for row in decode_rows:
            fh.write(json.dumps(row) + "\n")
    print(f"[{args.variant}] DECODE_OK {len(decode_rows)} prompts x{args.gen_tokens} tok", flush=True)

    # TPS
    tps = measure_tps(llm, args.tps_tokens, args.tps_repeats)
    print(f"[{args.variant}] TPS {tps['tps_mean']:.2f} +/- {tps['tps_std']:.2f} "
          f"(n={args.tps_repeats}, {args.tps_tokens} tok)", flush=True)

    # PPL
    ppl = None
    if not args.skip_ppl:
        ppl_recs = load_ppl_records(args.n_ppl)
        ppl = compute_ppl(llm, ppl_recs)
        print(f"[{args.variant}] PPL {ppl['ppl']:.4f} over {ppl['num_records']} recs "
              f"/ {ppl['num_tokens']} tok", flush=True)

    import torch
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    summary = {
        "variant": args.variant,
        "model": MODEL_ID,
        "load_s": load_s,
        "backend_summary": backend_summary,
        "tps": tps,
        "ppl": {k: v for k, v in (ppl or {}).items() if k != "per_record"} if ppl else None,
        "peak_mem_gb": peak_gb,
        "n_identity": args.n_identity,
        "gen_tokens": args.gen_tokens,
        "prefix_cache": not args.no_prefix_cache,
        "sequential": args.sequential,
        "enforce_eager": args.enforce_eager,
        "determinism": det,
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[{args.variant}] DONE peak_mem={peak_gb:.2f}GB -> {outdir}/summary.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
