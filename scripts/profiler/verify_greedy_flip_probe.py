#!/usr/bin/env python
"""Measure the int4-Marlin M=1 (AR greedy) vs M=K+1 (batched verify) per-token
argmax flip rate — the "linchpin" that gates speculative-decoding drafters on the
int4 QAT base — and test whether two cheap mechanism-targeted fixes remove it:

  * fp32-logit     : recompute the lm_head projection in fp32 (fp32 accumulation /
                     no bf16 output rounding) instead of the bf16 path.
  * deterministic  : torch.use_deterministic_algorithms(True) + CUBLAS_WORKSPACE_CONFIG=:4096:8
  * fp32-plus-det  : both at once.

Design (full write-up in research/verify_flip_probe/report.md)
--------------------------------------------------------------
The flip is driven by the GEMM batch dimension M (split-K reduction order in the
Marlin int4 kernel depends on M).  We isolate M cleanly:

  * Real int4 Marlin engine (vLLM 0.22, compressed-tensors W4A16, TRITON_ATTN).
  * enforce_eager=True so a chunked-prefill chunk of width M runs the decoder GEMMs
    at batch-dim exactly M (cudagraph capture would pad M=6 -> 8 and alias the sweep).
  * max_num_batched_tokens = M  =>  prompt_logprobs forces a recompute whose every
    interior position's logit is produced at GEMM batch-dim = M.  The rank-1 token of
    prompt_logprobs[c+1] is the *forced* argmax of the logit at position c given the
    real context S[:c] (forced decoding; M=1 and M=K+1 attend to the identical causal
    context — only the GEMM batch shape differs).
  * M=1 (max_num_batched_tokens=1) is the AR-greedy reference; M=K+1 are verify forwards.
  * flip(M) = fraction of (context, position) where argmax(M=1) != argmax(M=K+1).

Process model
-------------
vLLM V1's in-process engine teardown does NOT release GPU memory, so we cannot build
multiple max_num_batched_tokens engines in one process.  Each (config-family, M) runs
in its own fresh worker subprocess; the orchestrator collects per-position argmax maps
and computes flips across them.  Because the controlled path (eager, fixed shapes, no
Marlin atomic-add, seed=0) is process-to-process deterministic, cross-process M=1-vs-
M=K+1 isolates only the batch-shape effect; we VERIFY this by measuring a noise floor
(two independent M=1 processes — expected 0 flips) and report it.

deterministic needs CUBLAS_WORKSPACE_CONFIG set before the cuBLAS handle is created, so
{baseline, fp32-logit} workers run with no special env and {deterministic, fp32-plus-det}
workers run with the env preset + torch.use_deterministic_algorithms(True).

LOCAL ONLY — no HF Job, no submission change.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# config-family mapping
#   group A (no special env) : plain -> baseline,      fp32 -> fp32-logit
#   group B (CUBLAS det env) : plain -> deterministic, fp32 -> fp32-plus-det
FAMILY = {
    "A": {"plain": "baseline", "fp32": "fp32-logit", "deterministic": False},
    "B": {"plain": "deterministic", "fp32": "fp32-plus-det", "deterministic": True},
}
CONFIG_TO_GROUP = {
    "baseline": ("A", "plain"),
    "fp32-logit": ("A", "fp32"),
    "deterministic": ("B", "plain"),
    "fp32-plus-det": ("B", "fp32"),
}
ALL_CONFIGS = ["baseline", "fp32-logit", "deterministic", "fp32-plus-det"]
CONTEXT_CANDIDATES = [
    "research/local_validation/vllm_baseline/decode_outputs_128.jsonl",
    "research/greedy_reference/google__gemma-4-E4B-it/decode_outputs.jsonl",
]


# --------------------------------------------------------------------------- #
# shared: deterministic position selection (orchestrator + worker must agree)
# --------------------------------------------------------------------------- #
def load_contexts(path, num_contexts):
    records = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if len(records) >= num_contexts:
                break
    contexts = [list(r["prompt_token_ids"]) + list(r["completion_token_ids"]) for r in records]
    prompt_len = len(records[0]["prompt_token_ids"])
    return contexts, prompt_len


def positions_for(contexts, prompt_len, num_steps, max_k, block=4):
    """num_steps positions spaced `block` apart, starting right after the prompt.

    A SHALLOW, DENSE window (first `block * num_steps` completion tokens) instead of
    spreading positions to the sequence tail: at mnbt=1 every interior position costs
    one prefill forward, so the dominant cost is the *depth* of the deepest measured
    position. Packing positions densely just past the prompt keeps that depth ~minimal
    (prompt_len + block*num_steps) while still giving num_steps realistic decode-region
    positions per context. block>=2 keeps adjacent measured positions decorrelated."""
    start = (prompt_len // block) * block
    by_ctx = []
    for S in contexts:
        cap = len(S) - max_k - 2
        cand = [start + i * block for i in range(num_steps)]
        by_ctx.append([c for c in cand if block <= c <= cap])
    return by_ctx


# ============================================================================= #
#  WORKER  (one process: one env-group, one max_num_batched_tokens M)
# ============================================================================= #
def rank1_token(lp_dict):
    best_tok, best_lp = None, -1e30
    for tok, lp in lp_dict.items():
        val = lp.logprob if hasattr(lp, "logprob") else lp
        if val > best_lp:
            best_lp, best_tok = val, tok
    return best_tok


def lang_model(model):
    if hasattr(model, "logits_processor") and hasattr(model, "lm_head"):
        return model
    if hasattr(model, "language_model"):
        return model.language_model
    raise RuntimeError(f"cannot locate language model on {type(model).__name__}")


def get_model(llm):
    for path in (
        "llm.llm_engine.engine_core.engine_core.model_executor.driver_worker.model_runner.model",
        "llm.llm_engine.engine_core.model_executor.driver_worker.model_runner.model",
    ):
        try:
            return eval(path)  # noqa: S307 - trusted internal access
        except Exception:  # noqa: BLE001
            continue
    raise RuntimeError("could not reach in-process model (need VLLM_ENABLE_V1_MULTIPROCESSING=0)")


def install_fp32_logit_patch(model):
    """Recompute the lm_head projection in fp32 (fp32 accumulation, no bf16 output
    rounding). Overrides LogitsProcessor._get_logits only, so soft-cap/scale in
    LogitsProcessor.forward still run. Returns a restore() callable."""
    import torch

    lm = lang_model(model)
    lp = lm.logits_processor
    org_vocab = lp.org_vocab_size
    original = lp._get_logits

    def _get_logits_fp32(hidden_states, lm_head_arg, embedding_bias=None):
        logits = torch.nn.functional.linear(
            hidden_states.float(),
            lm_head_arg.weight.float(),
            embedding_bias.float() if embedding_bias is not None else None,
        )
        if logits is not None:
            logits = logits[..., :org_vocab]
        return logits

    lp._get_logits = _get_logits_fp32
    return lambda: setattr(lp, "_get_logits", original)


def build_engine(model_path, mnbt, max_model_len):
    from vllm import LLM

    return LLM(
        model=model_path, dtype="bfloat16", max_model_len=max_model_len,
        gpu_memory_utilization=0.88, max_num_batched_tokens=mnbt, max_num_seqs=1,
        enable_chunked_prefill=True, enable_prefix_caching=False, enforce_eager=True,
        trust_remote_code=True, disable_log_stats=True,
    )


def measure_argmaxes(llm, contexts, positions_by_ctx, logprobs_k, mnbt):
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=logprobs_k)
    out = {}
    for ci, (S, positions) in enumerate(zip(contexts, positions_by_ctx)):
        if not positions:
            continue
        # feed mnbt extra tokens past the deepest position so every measured position
        # sits in a FULL chunk of width mnbt (the last chunk may be a partial remainder,
        # which would compute that position at batch < mnbt and corrupt the M reading).
        P = min(len(S), max(positions) + mnbt + 2)
        res = llm.generate(TokensPrompt(prompt_token_ids=S[:P]), sp, use_tqdm=False)[0]
        plp = res.prompt_logprobs
        for c in positions:
            entry = plp[c + 1] if c + 1 < len(plp) else None
            if entry is not None:
                out[f"{ci}:{c}"] = rank1_token(entry)
    return out


def decode_latency_ms_per_token(llm, prompt_ids, n_tokens=100):
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    import torch

    llm.generate(TokensPrompt(prompt_token_ids=prompt_ids),
                 SamplingParams(temperature=0.0, max_tokens=4), use_tqdm=False)
    torch.cuda.synchronize()
    t0 = time.time()
    out = llm.generate(TokensPrompt(prompt_token_ids=prompt_ids),
                       SamplingParams(temperature=0.0, max_tokens=n_tokens), use_tqdm=False)[0]
    torch.cuda.synchronize()
    return 1000.0 * (time.time() - t0) / max(len(out.outputs[0].token_ids), 1)


def run_worker(args):
    import torch

    group = args.worker_group
    deterministic = FAMILY[group]["deterministic"]
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    contexts, prompt_len = load_contexts(args.contexts, args.num_contexts)
    positions = positions_for(contexts, prompt_len, args.num_steps, max(args.k_sweep), args.block)
    n_pos = sum(len(p) for p in positions)

    mnbt = args.mnbt
    print(f"[worker {group} M={mnbt}] building (det={deterministic}) ...", flush=True)
    t0 = time.time()
    llm = build_engine(args.int4_base, mnbt, args.max_model_len)
    model = get_model(llm)
    print(f"[worker {group} M={mnbt}] built in {time.time()-t0:.1f}s ({type(model).__name__}); "
          f"{n_pos} positions", flush=True)

    t0 = time.time()
    plain = measure_argmaxes(llm, contexts, positions, args.logprobs_k, mnbt)
    t_plain = time.time() - t0
    print(f"[worker {group} M={mnbt}] plain measured in {t_plain:.1f}s", flush=True)

    fp32 = None
    t_fp32 = None
    if args.measure_fp32:
        restore = install_fp32_logit_patch(model)
        t0 = time.time()
        fp32 = measure_argmaxes(llm, contexts, positions, args.logprobs_k, mnbt)
        t_fp32 = time.time() - t0
        restore()
        print(f"[worker {group} M={mnbt}] fp32 measured in {t_fp32:.1f}s", flush=True)

    latency = {}
    if args.bench_latency:
        bench_prompt = contexts[0][:prompt_len]
        try:
            latency["plain"] = decode_latency_ms_per_token(llm, bench_prompt)
            if args.measure_fp32:
                restore = install_fp32_logit_patch(model)
                latency["fp32"] = decode_latency_ms_per_token(llm, bench_prompt)
                restore()
        except Exception as exc:  # noqa: BLE001
            print(f"[worker {group} M={mnbt}] latency bench skipped: {exc!r}", flush=True)

    Path(args.worker_out).write_text(json.dumps({
        "group": group, "mnbt": mnbt, "deterministic": deterministic,
        "n_positions": n_pos, "t_plain_s": t_plain, "t_fp32_s": t_fp32,
        "latency_ms_per_token": latency,
        "plain": plain, "fp32": fp32,
    }))
    print(f"[worker {group} M={mnbt}] wrote {args.worker_out}", flush=True)


# ============================================================================= #
#  ORCHESTRATOR  (no torch; spawns workers, aggregates, logs)
# ============================================================================= #
def resolve_contexts(root, override):
    if override:
        return override
    for cand in CONTEXT_CANDIDATES:
        if (root / cand).exists():
            return str(root / cand)
    raise FileNotFoundError(f"no decode-context jsonl among {CONTEXT_CANDIDATES}")


def _cached_worker_ok(out_path, group, mnbt, measure_fp32, expected_keys):
    """A cached worker file may be reused only if it was produced for the SAME
    (group, mnbt) AND over the EXACT same position-key set we would recompute now.
    Otherwise flip_stats would silently intersect mismatched keys."""
    try:
        d = json.loads(Path(out_path).read_text())
    except Exception:  # noqa: BLE001
        return None
    if d.get("group") != group or d.get("mnbt") != mnbt:
        return None
    if measure_fp32 and d.get("fp32") is None:
        return None
    if set((d.get("plain") or {}).keys()) != expected_keys:
        return None
    return d


def spawn_worker(root, group, mnbt, measure_fp32, args, contexts_path, out_path, bench_latency, tag,
                 resume=False, expected_keys=None):
    if resume and expected_keys is not None:
        cached = _cached_worker_ok(out_path, group, mnbt, measure_fp32, expected_keys)
        if cached is not None:
            print(f"[orchestrator] resume {tag}: reuse {out_path} "
                  f"({len(cached.get('plain') or {})} positions, no respawn)", flush=True)
            return cached
    env = dict(os.environ)
    env.update({
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0", "HF_HUB_OFFLINE": "1",
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "CUDA_VISIBLE_DEVICES": env.get("CUDA_VISIBLE_DEVICES", "0"),
    })
    if FAMILY[group]["deterministic"]:
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    cmd = [
        sys.executable, str(Path(__file__).resolve()),
        "--worker-group", group, "--mnbt", str(mnbt),
        "--measure-fp32", "1" if measure_fp32 else "0",
        "--worker-out", str(out_path), "--int4-base", args.int4_base,
        "--contexts", contexts_path, "--k-sweep", ",".join(map(str, args.k_sweep)),
        "--num-contexts", str(args.num_contexts), "--num-steps", str(args.num_steps),
        "--max-model-len", str(args.max_model_len), "--logprobs-k", str(args.logprobs_k),
        "--block", str(args.block),
    ]
    if bench_latency:
        cmd.append("--bench-latency")
    print(f"[orchestrator] spawn {tag}: group={group} M={mnbt} fp32={measure_fp32}", flush=True)
    subprocess.run(cmd, env=env, check=True, cwd=str(root))
    return json.loads(Path(out_path).read_text())


def flip_stats(ref, cand):
    keys = set(ref) & set(cand)
    flips = sum(1 for k in keys if ref[k] != cand[k])
    total = len(keys)
    return {"flips": flips, "total": total,
            "flip_rate_per_token": (flips / total) if total else None}


def run_orchestrator(args):
    root = Path(__file__).resolve().parents[2]
    contexts_path = resolve_contexts(root, args.contexts)
    out_path = (root / args.output) if not os.path.isabs(args.output) else Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.parent

    groups = sorted({CONFIG_TO_GROUP[c][0] for c in args.configs})
    Ms = [k + 1 for k in args.k_sweep]
    mnbts = sorted({1} | set(Ms))

    # position keys a fresh worker would produce now (torch-free); a cached worker
    # file may only be reused under --resume if its keys match this exactly.
    exp_contexts, exp_plen = load_contexts(contexts_path, args.num_contexts)
    exp_positions = positions_for(exp_contexts, exp_plen, args.num_steps, max(args.k_sweep), args.block)
    expected_keys = {f"{ci}:{c}" for ci, ps in enumerate(exp_positions) for c in ps}
    print(f"[orchestrator] expecting {len(expected_keys)} positions "
          f"({args.num_contexts} contexts x {args.num_steps} steps, block {args.block})", flush=True)

    # collect per (group, mnbt) argmax maps
    data = {g: {} for g in groups}            # data[group][mnbt] = worker result
    latency = {g: {} for g in groups}
    for g in groups:
        measure_fp32 = any(CONFIG_TO_GROUP[c] == (g, "fp32") for c in args.configs)
        measure_plain = any(CONFIG_TO_GROUP[c] == (g, "plain") for c in args.configs)
        for mnbt in mnbts:
            bench = mnbt in (1, max(mnbts))
            res = spawn_worker(root, g, mnbt, measure_fp32, args, contexts_path,
                               tmp / f".w_{g}_M{mnbt}.json", bench, f"{g}/M{mnbt}",
                               resume=args.resume, expected_keys=expected_keys)
            data[g][mnbt] = res
            if res.get("latency_ms_per_token"):
                latency[g][mnbt] = res["latency_ms_per_token"]
        # mark which variants this group needs
        data[g]["_need"] = {"plain": measure_plain, "fp32": measure_fp32}

    # noise floor: a second independent M=1 (group A, plain) vs the first
    noise = None
    if "A" in groups:
        res2 = spawn_worker(root, "A", 1, False, args, contexts_path,
                            tmp / ".w_A_M1_noise.json", False, "A/M1-noise")
        noise = flip_stats(data["A"][1]["plain"], res2["plain"])

    # flips per config
    configs_out = {}
    for cfg in args.configs:
        g, variant = CONFIG_TO_GROUP[cfg]
        ref = data[g][1][variant]
        per_m = {str(k + 1): flip_stats(ref, data[g][k + 1][variant]) for k in args.k_sweep}
        configs_out[cfg] = {"k_sweep": args.k_sweep, "ref_positions": len(ref), "per_M": per_m}

    overhead = compute_latency_overhead(latency, Ms)
    merged = {
        "configs": configs_out, "latency_overhead_pct": overhead,
        "latency_ms_per_token": latency, "noise_floor_M1_vs_M1": noise,
        "meta": {
            "int4_base": args.int4_base, "contexts_file": contexts_path,
            "num_contexts": args.num_contexts, "num_steps": args.num_steps,
            "k_sweep": args.k_sweep, "M_values": Ms, "configs": args.configs,
            "enforce_eager": True, "max_model_len": args.max_model_len,
        },
    }
    out_path.write_text(json.dumps(merged, indent=2))
    print(f"[orchestrator] wrote {out_path}", flush=True)

    report = build_report(merged, args)
    (out_path.parent / "report.md").write_text(report)
    print(f"[orchestrator] wrote {out_path.parent / 'report.md'}", flush=True)
    _maybe_log_wandb(root, args, merged)
    print("\n" + report, flush=True)


def compute_latency_overhead(latency, Ms):
    """fp32 overhead within-process; det overhead vs group-A baseline (cross-process)."""
    mverify = max(Ms) if Ms else 8

    def get(g, mnbt, variant):
        return latency.get(g, {}).get(mnbt, {}).get(variant)

    base = get("A", mverify, "plain") or get("A", 1, "plain")
    overhead = {"baseline": 0.0}
    if base:
        fa = get("A", mverify, "fp32")
        if fa is not None:
            overhead["fp32-logit"] = 100.0 * (fa - base) / base
        db = get("B", mverify, "plain")
        if db is not None:
            overhead["deterministic"] = 100.0 * (db - base) / base
        dbf = get("B", mverify, "fp32")
        if dbf is not None:
            overhead["fp32-plus-det"] = 100.0 * (dbf - base) / base
    return overhead


def build_report(results, args):
    Ms = [k + 1 for k in args.k_sweep]
    header = "| config | " + " | ".join(f"flip_rate (M={M})" for M in Ms) + " | latency overhead |"
    sep = "|" + "---|" * (len(Ms) + 2)
    rows = []
    for cfg in args.configs:
        cells = []
        per_m = results["configs"].get(cfg, {}).get("per_M", {})
        for M in Ms:
            st = per_m.get(str(M), {})
            r = st.get("flip_rate_per_token")
            cells.append(f"{r:.5f} ({st.get('flips')}/{st.get('total')})" if r is not None else "n/a")
        ov = results["latency_overhead_pct"].get(cfg)
        ov_str = "0%" if cfg == "baseline" else (f"{ov:+.1f}%" if ov is not None else "n/a")
        rows.append(f"| {cfg} | " + " | ".join(cells) + f" | {ov_str} |")
    table = "\n".join([header, sep, *rows])

    def rate(cfg, M):
        return results["configs"].get(cfg, {}).get("per_M", {}).get(str(M), {}).get("flip_rate_per_token")

    zero = [cfg for cfg in args.configs if all(rate(cfg, M) == 0.0 for M in Ms)]
    reduced = [cfg for cfg in args.configs if cfg != "baseline"
               and any((rate(cfg, M) or 0) < (rate("baseline", M) or 0) for M in Ms)]
    if zero:
        decision = (f"**LINCHPIN CANDIDATE FOUND** — {zero} reach flip_rate_per_token = 0.000 across "
                    f"M ∈ {Ms}. Evaluate latency overhead vs kanna #19's batch-invariant kernel.")
    elif reduced:
        decision = (f"**PARTIAL** — {reduced} reduce but do not eliminate the flip; useful calibration for "
                    "the batch-invariant-kernel work, not a standalone linchpin resolution.")
    else:
        decision = ("**CHEAP FIXES RULED OUT** — fp32 accumulation and deterministic reduction leave the flip "
                    "rate unchanged; the root cause is the batch-tiling structure itself, so a batch-invariant "
                    "kernel (padding M to a fixed tile) is the required fix.")

    nf = results.get("noise_floor_M1_vs_M1")
    nf_str = (f"{nf['flips']}/{nf['total']} (rate {nf['flip_rate_per_token']:.5f})"
              if nf else "not measured")
    return f"""# int4 spec-verify greedy flip-rate probe (PR #23)

Per-token argmax flip between **M=1 (AR greedy)** and **M=K+1 (batched verify)** on the
merged int4 QAT base (`{args.int4_base}`), and whether fp32-logit accumulation or
deterministic reduction removes it.

## Result table

flip_rate_per_token = flips / (contexts × positions); cell shows `rate (flips/total)`.

{table}

- contexts = {args.num_contexts}, positions/context = {args.num_steps} (spaced {args.block} tokens apart,
  starting right after the {{prompt_len}}-token prompt — a shallow dense decode-region window), k-sweep =
  {args.k_sweep} (M = K+1). Total compared (context, position) pairs per config ≈ contexts × positions.
- **Cross-process determinism noise floor** (two independent M=1 runs): {nf_str}. A ~0 floor means the
  flips above are the genuine batch-shape (M) effect, not process-boundary noise.
- Latency overhead = single-sequence decode ms/token vs `baseline` at the verify size (M={max(Ms)}).

## Decision

{decision}

## Method (why this isolates the GEMM batch dimension M)

- Real int4 Marlin engine (vLLM 0.22, compressed-tensors W4A16, TRITON_ATTN), `enforce_eager=True`
  so a chunked-prefill chunk of width M runs the decoder GEMMs at batch-dim exactly M.
- `max_num_batched_tokens = M`; `prompt_logprobs` recomputes every interior position's logit at
  batch-dim = M. The rank-1 token of `prompt_logprobs[c+1]` is the forced argmax of the logit at
  position c given the real context S[:c].
- M=1 (`max_num_batched_tokens=1`) is the AR-greedy reference; M=K+1 are the verify forwards, both
  forced on the identical causal context — only the GEMM batch shape differs.
- vLLM V1 in-process teardown does not free GPU memory, so each (config-family, M) runs in its own
  fresh worker subprocess. The path is process-to-process deterministic (eager, fixed shapes, no
  Marlin atomic-add, seed=0), verified by the noise floor above, so cross-process comparison isolates
  only the batch-shape effect.
- deterministic configs run with `CUBLAS_WORKSPACE_CONFIG=:4096:8` (set before the cuBLAS handle is
  created) and `torch.use_deterministic_algorithms(True, warn_only=True)`.

### fp32-logit semantics
`fp32-logit` recomputes the lm_head projection in fp32 (`F.linear(hidden.float(), weight.float())`),
i.e. fp32 accumulation with no bf16 output rounding. The trivial `logits.to(float32)` cast is a
mathematical no-op for argmax (monotone, and the Gemma final-logit soft-cap is also monotone), so the
fp32-accumulation form is the only faithful test of hypothesis (a).

## Caveats
- Deterministic-mode overhead is measured cross-process (group A vs group B); treat as approximate.
  fp32-logit overhead is within-process (patch on/off, same engine).
- `torch.use_deterministic_algorithms` / `CUBLAS_WORKSPACE_CONFIG` do not alter the custom Marlin int4
  CUDA kernel or the Triton attention kernel; they only constrain cuBLAS/cuDNN/torch ops.
"""


def _maybe_log_wandb(root, args, merged):
    if args.no_wandb:
        return
    try:
        sys.path.insert(0, str(root))
        from scripts.wandb_logging import DEFAULT_WANDB_ENTITY, DEFAULT_WANDB_PROJECT
        import wandb
    except Exception as exc:  # noqa: BLE001
        print(f"[orchestrator] W&B disabled ({exc!r})", flush=True)
        return
    if not (os.environ.get("WANDB_API_KEY") or os.environ.get("WANDB_MODE")):
        print("[orchestrator] no WANDB_API_KEY/WANDB_MODE; skipping W&B", flush=True)
        return
    Ms = merged["meta"]["M_values"]
    try:
        run = wandb.init(
            entity=os.environ.get("WANDB_ENTITY") or DEFAULT_WANDB_ENTITY,
            project=os.environ.get("WANDB_PROJECT") or DEFAULT_WANDB_PROJECT,
            group=args.wandb_group, name=args.wandb_name, job_type="flip-probe",
            tags=["gemma-challenge", "linchpin", "flip-rate"], config=merged["meta"],
        )
        # per-M flip_rate curves (one series per config), x-axis = M
        for cfg, d in merged["configs"].items():
            for M_str, st in sorted(d.get("per_M", {}).items(), key=lambda kv: int(kv[0])):
                if st.get("flip_rate_per_token") is not None:
                    run.log({"M": int(M_str), f"flip_rate/{cfg}": st["flip_rate_per_token"],
                             f"flips/{cfg}": st["flips"], "global_step": int(M_str)})
        # full flip grid as a table
        table = wandb.Table(columns=["config", "M", "flips", "total", "flip_rate_per_token"])
        for cfg in merged["meta"]["configs"]:
            for M_str, st in sorted(merged["configs"].get(cfg, {}).get("per_M", {}).items(),
                                    key=lambda kv: int(kv[0])):
                table.add_data(cfg, int(M_str), st.get("flips"), st.get("total"),
                               st.get("flip_rate_per_token"))
        run.log({"flip_grid": table})
        # headline summaries
        for cfg, ov in merged["latency_overhead_pct"].items():
            run.summary[f"latency_overhead_pct/{cfg}"] = ov
        for cfg, d in merged["configs"].items():
            for M_str, st in d.get("per_M", {}).items():
                run.summary[f"flip_rate/{cfg}/M{M_str}"] = st.get("flip_rate_per_token")
        if merged.get("noise_floor_M1_vs_M1"):
            run.summary["noise_floor/flip_rate"] = merged["noise_floor_M1_vs_M1"]["flip_rate_per_token"]

        def _all_zero(cfg):
            pm = merged["configs"].get(cfg, {}).get("per_M", {})
            return bool(pm) and all(pm.get(str(M), {}).get("flip_rate_per_token") == 0.0 for M in Ms)

        zero_cfgs = [c for c in merged["meta"]["configs"] if _all_zero(c)]
        run.summary["zero_flip_configs"] = ",".join(zero_cfgs) if zero_cfgs else "none"
        base_rates = [merged["configs"].get("baseline", {}).get("per_M", {}).get(str(M), {})
                      .get("flip_rate_per_token") for M in Ms]
        base_rates = [r for r in base_rates if r is not None]
        run.summary["baseline/max_flip_rate"] = max(base_rates) if base_rates else None
        run.finish()
        print(f"[orchestrator] logged to W&B (zero-flip configs: {zero_cfgs or 'none'})", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[orchestrator] W&B logging failed ({exc!r}); results.json/report.md already written",
              flush=True)


# ============================================================================= #
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--int4-base", default="google/gemma-4-E4B-it-qat-w4a16-ct")
    p.add_argument("--k-sweep", default="1,3,5,7", help="comma list of K; M = K+1")
    p.add_argument("--num-contexts", type=int, default=64)
    p.add_argument("--num-steps", type=int, default=32)
    p.add_argument("--block", type=int, default=4, help="spacing between measured positions")
    p.add_argument("--configs", default=",".join(ALL_CONFIGS))
    p.add_argument("--output", default="research/verify_flip_probe/results.json")
    p.add_argument("--contexts", default=None)
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--logprobs-k", type=int, default=5)
    p.add_argument("--wandb-group", default="verify-greedy-flip-probe")
    p.add_argument("--wandb-name", default="stark/linchpin-fp32-accum-flip-probe")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--resume", action="store_true",
                   help="reuse cached .w_<group>_M<mnbt>.json workers whose position keys match")
    # worker-mode (internal)
    p.add_argument("--worker-group", default=None, choices=[None, "A", "B"])
    p.add_argument("--mnbt", type=int, default=None)
    p.add_argument("--worker-out", default=None)
    p.add_argument("--measure-fp32", type=int, default=0)
    p.add_argument("--bench-latency", action="store_true")
    a = p.parse_args()
    a.k_sweep = [int(x) for x in str(a.k_sweep).split(",") if x.strip()]
    a.configs = [c.strip() for c in str(a.configs).split(",") if c.strip()]
    for c in a.configs:
        if c not in CONFIG_TO_GROUP:
            raise SystemExit(f"unknown config {c!r}; choose from {ALL_CONFIGS}")
    return a


def main():
    args = parse_args()
    if args.worker_group:
        run_worker(args)
    else:
        run_orchestrator(args)


if __name__ == "__main__":
    main()
