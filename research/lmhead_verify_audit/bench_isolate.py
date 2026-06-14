"""Isolate the lm_head verify GEMM (the lever target: 12288-col read) from the
262144-scatter, so the reclaimable share is honest. Same faithful load as bench_lmhead_verify.py.
"""
import os, sys, json, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
SUB = "/workspace/senpai/target/submissions/fa2sw_precache_kenyan"
sys.path.insert(0, SUB)
os.environ["PCK04_KEEPSET"] = "/tmp/osoi5-12k-baked/pck04_keepset.json"
import serve_patch_pck04  # noqa: F401
import torch
from vllm import LLM

HIDDEN, K_HEAD, FULL_VOCAB, M = 2560, 12288, 262144, 8


def bench(fn, iters=300, warmup=80):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000  # us


def main():
    llm = LLM(model="/tmp/osoi5-12k-baked", quantization="compressed-tensors", dtype="bfloat16",
              max_model_len=4096, gpu_memory_utilization=0.90, max_num_seqs=1,
              enforce_eager=True, trust_remote_code=True)
    mr = llm.llm_engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner
    model = mr.model
    # locate the 12288-row ParallelLMHead
    lmh = None
    for name, mod in model.named_modules():
        if name.endswith("lm_head") and hasattr(mod, "quant_method"):
            lmh = mod; lmh_name = name; break
    print(f"[iso] lm_head module = {lmh_name} ({type(lmh).__name__}); "
          f"weight_packed={tuple(lmh.weight_packed.shape)}", flush=True)

    dev = lmh.weight_packed.device
    h = torch.randn(M, HIDDEN, dtype=torch.bfloat16, device=dev)

    # 1) GEMM only: the real int4 Marlin apply -> [M, 12288]  (THE LEVER TARGET)
    qm = lmh.quant_method
    with torch.inference_mode():
        g = qm.apply(lmh, h, bias=None)
        assert g.shape[-1] == K_HEAD, g.shape
        t_gemm = bench(lambda: qm.apply(lmh, h, bias=None))
    print(f"[iso] int4 Marlin GEMM only  [8x2560]@[2560x12288] -> [8,12288] : {t_gemm:.2f} us", flush=True)

    # 2) scatter only: index_copy_ [M,12288] -> cached [M,262144]
    out = torch.full((M, FULL_VOCAB), float("-inf"), dtype=g.dtype, device=dev)
    keep_idx = torch.arange(K_HEAD, device=dev)
    pruned = g.contiguous()
    def scatter_only():
        out.index_copy_(1, keep_idx, pruned)
    with torch.inference_mode():
        t_scatter = bench(scatter_only)
    print(f"[iso] scatter only  index_copy_ [8,12288]->[8,262144]            : {t_scatter:.2f} us", flush=True)

    # 3) per-row argmax over candidate set (263) — what a candidate verify would do instead of scatter+argmax
    cand = 263
    w_bf16 = torch.randn(K_HEAD, HIDDEN, dtype=torch.bfloat16, device=dev)
    cand_idx = torch.randint(0, K_HEAD, (M, cand), device=dev)
    def cand_path():
        emb = w_bf16[cand_idx.reshape(-1)].view(M, cand, HIDDEN)
        logits = torch.einsum("md,mcd->mc", h, emb)
        return logits.argmax(-1)
    with torch.inference_mode():
        t_cand = bench(cand_path)
    print(f"[iso] candidate per-row gather-GEMM+argmax (8x263, bf16 UB)      : {t_cand:.2f} us", flush=True)

    res = {"gemm_only_us": t_gemm, "scatter_only_us": t_scatter,
           "cand_perrow_argmax_us": t_cand, "compute_logits_full_us_prev": 135.82}
    print("ISO_JSON " + json.dumps(res), flush=True)
    with open("/workspace/senpai/target/research/lmhead_verify_audit/iso_result.json", "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
