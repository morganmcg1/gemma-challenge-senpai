"""AUDIT (PR #144): quantify the verify-step lm_head GEMM share + dense-vs-candidate cost.

Loads the SERVED 12288-row int4 pruned lm_head (/tmp/osoi5-12k-baked) through the
real PCK-04-patched compute_logits path and times it at the verify shape M=K+1=8.
Then times a bf16 candidate-restricted (263-row) gather-GEMM as an OPTIMISTIC upper
bound on any "active vocabulary" verify shortcut (no int4 unpack, no correctness
certificate). Decisive question: is the lm_head a reclaimable share, and can a
263-col read beat the dense 12288 int4 GEMM at M=8?

Single A10G, LOCAL profiling only. No HF Job. Greedy untouched.
"""
import os, sys, json, time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

SUB = "/workspace/senpai/target/submissions/fa2sw_precache_kenyan"
sys.path.insert(0, SUB)
os.environ["PCK04_KEEPSET"] = "/tmp/osoi5-12k-baked/pck04_keepset.json"
import serve_patch_pck04  # noqa: F401  (registers gemma4 finder: rebuild head K=12288 + patch compute_logits)

import torch
from vllm import LLM

MODEL = "/tmp/osoi5-12k-baked"
HIDDEN = 2560
K_HEAD = 12288
FULL_VOCAB = 262144
K_SPEC = 7          # num_speculative_tokens
M_VERIFY = K_SPEC + 1  # verify rows per step = 8
CAND = 263          # {drafter K=7 ∪ top-256 bonus}


def bench(fn, iters=300, warmup=80):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters  # ms


def get_model(llm):
    paths = [
        lambda: llm.llm_engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner.model,
        lambda: llm.llm_engine.model_executor.driver_worker.worker.model_runner.model,
        lambda: llm.llm_engine.model_executor.driver_worker.model_runner.model,
    ]
    for p in paths:
        try:
            m = p()
            if m is not None:
                return m
        except Exception:
            continue
    raise RuntimeError("could not locate model_runner.model")


def main():
    print("[bench] loading served 12288-head model via vLLM ...", flush=True)
    t0 = time.time()
    llm = LLM(
        model=MODEL,
        quantization="compressed-tensors",
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.90,
        max_num_seqs=1,
        enforce_eager=True,
        trust_remote_code=True,
    )
    print(f"[bench] load done in {time.time()-t0:.0f}s", flush=True)

    model = get_model(llm)
    dev = next(model.parameters()).device
    print(f"[bench] model on {dev}; lm_head={type(getattr(model,'lm_head',None)).__name__}", flush=True)

    res = {"device": str(dev), "K_head": K_HEAD, "full_vocab": FULL_VOCAB,
           "M_verify": M_VERIFY, "cand": CAND, "compute_logits_us": {}}

    # --- faithful: real PCK-04 compute_logits (int4 Marlin GEMM over 12288 rows + scatter to 262144) ---
    for M in [1, M_VERIFY, 16, 32]:
        h = torch.randn(M, HIDDEN, dtype=torch.bfloat16, device=dev)
        try:
            out = model.compute_logits(h)
            assert out is not None and out.shape[-1] == FULL_VOCAB, f"unexpected out {None if out is None else out.shape}"
            t = bench(lambda: model.compute_logits(h))
            res["compute_logits_us"][M] = t * 1000
            print(f"[bench] compute_logits(int4-12288 + scatter) M={M}: {t*1000:.2f} us  out={tuple(out.shape)}", flush=True)
        except Exception as exc:
            res["compute_logits_us"][M] = f"ERR {exc!r}"
            print(f"[bench] compute_logits M={M} FAILED: {exc!r}", flush=True)

    # --- candidate-restricted UPPER BOUND: bf16 263-row gather-GEMM at M=8 (no int4 unpack, no cert) ---
    wfull_bf16 = torch.randn(K_HEAD, HIDDEN, dtype=torch.bfloat16, device=dev)  # dummy weights; timing only
    h8 = torch.randn(M_VERIFY, HIDDEN, dtype=torch.bfloat16, device=dev)
    cand_idx = torch.randint(0, K_HEAD, (M_VERIFY, CAND), device=dev)  # per-row distinct candidates

    def sparse_per_row_gather_gemm():
        # per verify row, gather CAND rows then dot: faithful to per-row {K∪bonus}
        emb = wfull_bf16[cand_idx.reshape(-1)].view(M_VERIFY, CAND, HIDDEN)
        return torch.einsum("md,mcd->mc", h8, emb)

    shared_idx = torch.randint(0, K_HEAD, (CAND,), device=dev)
    def sparse_shared_gemm():
        emb = wfull_bf16[shared_idx]            # [CAND, HIDDEN]
        return h8 @ emb.t()                     # [8, CAND]

    def dense_bf16_ref():
        return h8 @ wfull_bf16.t()              # [8, 12288] bf16 dense (no int4)

    res["sparse_perrow_us"] = bench(sparse_per_row_gather_gemm) * 1000
    res["sparse_shared_us"] = bench(sparse_shared_gemm) * 1000
    res["dense_bf16_ref_us"] = bench(dense_bf16_ref) * 1000
    print(f"[bench] sparse per-row gather-GEMM (8x263, bf16 UPPER BOUND): {res['sparse_perrow_us']:.2f} us", flush=True)
    print(f"[bench] sparse shared gather-GEMM (8x263, bf16):              {res['sparse_shared_us']:.2f} us", flush=True)
    print(f"[bench] dense bf16 ref GEMM (8x12288):                        {res['dense_bf16_ref_us']:.2f} us", flush=True)

    # --- shares vs per-accepted-token budget at baseline 481.53 TPS ---
    tok_budget_us = 1e6 / 481.53
    cl8 = res["compute_logits_us"].get(M_VERIFY)
    if isinstance(cl8, (int, float)):
        res["lmhead_verify_us"] = cl8
        res["per_token_budget_us"] = tok_budget_us
        for ET in [2.5, 3.0, 3.5, 4.0]:
            step_us = ET * tok_budget_us
            res[f"share_of_step_ET{ET}"] = cl8 / step_us
        print(f"[bench] per-accepted-token budget @481.53 TPS = {tok_budget_us:.1f} us", flush=True)
        for ET in [2.5, 3.0, 3.5, 4.0]:
            print(f"[bench]   lm_head verify share of decode step (E[T]={ET}): {100*cl8/(ET*tok_budget_us):.3f}%", flush=True)

    with open("/workspace/senpai/target/research/lmhead_verify_audit/bench_result.json", "w") as f:
        json.dump(res, f, indent=2, default=str)
    print("[bench] wrote bench_result.json", flush=True)
    print("RESULT_JSON " + json.dumps(res, default=str), flush=True)


if __name__ == "__main__":
    main()
