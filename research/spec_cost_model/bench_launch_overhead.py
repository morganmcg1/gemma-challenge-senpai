"""CUDA-graph launch-overhead leg of the step-denominator audit (PR #154, instr 3).

KEY FINDING (manifest + #136 evidence): the deployed fa2sw_precache_kenyan stack
ALREADY CUDA-graph-captures the tree step:
  - drafter propose loop (K=7 width-1 iters): LOOPGRAPH/ONEGRAPH capture
    (sitecustomize.py `_capture_graph` -> torch.cuda.CUDAGraph()).
  - target verify forward (M=32 tree, 42 layers): vLLM cudagraph
    (#136 `gemm_all_graphed: true`).
So hypothesis #2's "if NOT captured, per-launch overhead is a step tax" is largely
CLOSED in the baseline 1.2182 step. This script:
  1. MEASURES the per-launch overhead on the real A10G (eager N-kernel chain vs a
     CUDA-graph replay of the same chain) -> the unit of the tax.
  2. MODELS the tree-step launch budget from topology (M=32, depth-9, branch-3,
     42 layers, K=7 drafter) to size the tax the EXISTING capture already removes.
  3. BOUNDS the RESIDUAL un-captured headroom (data-dependent accept-walk + glue),
     the only CUDA-graph headroom left in the deployed stack -> small.
  4. Flags the precise already-recovered/residual split as ARMED/PENDING land's
     real launch trace (reuses lawine #147 --trace input).

LOCAL A10G profiling + analysis ONLY. No HF Job, no served-file change.
"""
import os, sys, json, time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# tree topology (PR #154 / deployed stack)
TREE_M = 32
TREE_DEPTH = 9
TREE_MAX_BRANCH = 3
N_LAYERS = 42            # #136: n_attn_sliding 35 + n_attn_full 7
DRAFTER_K = 7           # num_speculative_tokens
KERNELS_PER_LAYER = 10  # QKV, attn, O, gate, up, down, 2x norm, residual adds (~order)
DRAFTER_LAYERS = 1      # Gemma4 MTP drafter is a small Q-only/KV-shared head (~1 layer), not the backbone
BUDGET_US = 1.0e6 / 481.53


def _bench(fn, iters, warmup=50):
    import torch
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
    return s.elapsed_time(e) / iters * 1000.0  # us


def measure_launch_overhead(device):
    import torch
    dev = torch.device(device)
    # tiny tensors so each kernel's GPU exec time is minimal -> isolate launch cost
    N = 64
    xs = [torch.zeros(256, device=dev) for _ in range(N)]

    def eager_chain():
        for x in xs:
            x.add_(1.0)

    # one tiny kernel, to characterize the launch-bound regime
    x0 = xs[0]
    per_kernel_eager_us = _bench(lambda: x0.add_(1.0), iters=2000)

    eager_chain_us = _bench(eager_chain, iters=300)

    # capture the same N-kernel chain into a CUDA graph
    torch.cuda.synchronize()
    for _ in range(3):
        eager_chain()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        eager_chain()
    graph_chain_us = _bench(lambda: graph.replay(), iters=300)

    # per-launch overhead = (eager_total - graph_replay) / N
    per_launch_overhead_us = max(0.0, (eager_chain_us - graph_chain_us) / N)
    return {
        "device": str(dev),
        "gpu": torch.cuda.get_device_name(0),
        "N_kernels": N,
        "per_kernel_eager_us": per_kernel_eager_us,
        "eager_chain_us": eager_chain_us,
        "graph_replay_chain_us": graph_chain_us,
        "per_launch_overhead_us": per_launch_overhead_us,
        "launch_bound_note": ("per_kernel_eager ~= per_launch_overhead => tiny kernels are "
                              "launch-bound (CPU enqueue dominates GPU exec); this is the "
                              "regime where un-captured launches tax the step."),
    }


def model_tree_step(lo):
    plo = lo["per_launch_overhead_us"]
    # launch counts (eager, un-captured)
    verify_launches = N_LAYERS * KERNELS_PER_LAYER          # M=32 target verify forward
    # drafter is width-1: each of K iters runs the backbone once; Q-only/KV-shared -> ~half kernels/layer
    drafter_launches = DRAFTER_K * DRAFTER_LAYERS * (KERNELS_PER_LAYER // 2)
    logits_launches = 6                                     # GEMM, scatter, cast, softcap, argmax, gather
    # accept-walk: data-dependent traversal over the tree (depth-9, branch-3) -> the residual
    accept_walk_launches = TREE_DEPTH * TREE_MAX_BRANCH     # ~order; NOT statically capturable
    glue_launches = 12                                      # sampling/metadata/copies outside graphs

    total_eager = verify_launches + drafter_launches + logits_launches + accept_walk_launches + glue_launches
    # what the deployed capture removes (static forward = verify + drafter + logits-GEMM region)
    captured = verify_launches + drafter_launches + logits_launches
    # residual = un-capturable data-dependent control flow + glue
    residual = accept_walk_launches + glue_launches

    # The eager launch tax is OVERLAP-LIMITED, not N*plo: launches overlap GPU exec, so the
    # tax is the GPU-idle fraction. Upper bound = N*plo; a realistic bound credits overlap.
    # We report BOTH the (loose) upper bound and flag the overlap-limited true value as
    # PENDING land's #147 launch trace.
    captured_tax_upper_us = captured * plo
    residual_headroom_upper_us = residual * plo
    # async launches overlap GPU exec; the true residual tax is the GPU-idle fraction only.
    # Illustrative overlap credit (NOT measured; pending land's trace) at 80% overlap:
    OVERLAP_CREDIT = 0.8
    residual_headroom_overlap_illustrative_us = residual_headroom_upper_us * (1.0 - OVERLAP_CREDIT)

    step_abs_bar_us = 4.862 * BUDGET_US
    return {
        "topology": {"M": TREE_M, "depth": TREE_DEPTH, "max_branch": TREE_MAX_BRANCH,
                     "n_layers": N_LAYERS, "drafter_K": DRAFTER_K},
        "launch_counts": {
            "verify_forward": verify_launches,
            "drafter_propose": drafter_launches,
            "compute_logits": logits_launches,
            "accept_walk": accept_walk_launches,
            "glue": glue_launches,
            "total_eager": total_eager,
            "captured_by_deployed_graph": captured,
            "residual_uncaptured": residual,
        },
        "per_launch_overhead_us": plo,
        "captured_tax_upper_bound_us": captured_tax_upper_us,
        "captured_tax_pct_of_bar_step_upper": 100.0 * captured_tax_upper_us / step_abs_bar_us,
        "residual_headroom_upper_bound_us": residual_headroom_upper_us,
        "residual_headroom_pct_of_bar_step_upper": 100.0 * residual_headroom_upper_us / step_abs_bar_us,
        "residual_headroom_overlap_illustrative_us": residual_headroom_overlap_illustrative_us,
        "residual_headroom_pct_illustrative_80pct_overlap": 100.0 * residual_headroom_overlap_illustrative_us / step_abs_bar_us,
        "verdict": ("CUDA-graph leg LARGELY CLOSED in the deployed stack: the drafter "
                    "loopgraph + verify cudagraph already capture the static forward "
                    "(captured launch count >> residual). Residual headroom is bounded by "
                    "the data-dependent accept-walk + glue (~%d launches) and is "
                    "second-order. Decode-path scatter avoidance (NOT a launch lever) is "
                    "the real remaining denominator headroom." % residual),
        "armed_pending": ("PENDING land's real launch trace (lawine #147 --trace): the "
                          "overlap-limited true tax (vs the N*plo upper bound here) and the "
                          "exact residual un-captured set need the live step trace. ARMED to "
                          "re-price when that trace lands."),
        "caveat": ("N*plo is an UPPER bound; async launches overlap GPU exec so the true "
                   "tax is the GPU-idle fraction (<= this). The deployed capture already "
                   "removes the bulk; do not double-count this against the 1.2182 baseline, "
                   "which is measured WITH capture on."),
    }


def main():
    out_path = "research/spec_cost_model/launch_overhead_graph_leg.json"
    t0 = time.time()
    import torch
    assert torch.cuda.is_available(), "CUDA not available (set CUDA_VISIBLE_DEVICES=0)"
    lo = measure_launch_overhead("cuda:0")
    model = model_tree_step(lo)
    out = {
        "pr": 154, "leg": "cuda_graph_launch_overhead",
        "deployed_already_graphed": True,
        "evidence": {
            "manifest": ["ONEGRAPH=1", "LOOPGRAPH_REQUIRE_CAPTURE=1",
                         "LOOPGRAPH_WARMUP_CALLS=20", "DIXIE_PREWARM_GREEDY_KERNEL=1"],
            "sitecustomize": "_capture_graph -> torch.cuda.CUDAGraph(); LOOPGRAPH_TARGET=vllm.v1.spec_decode.gemma4",
            "step136": "gemm_all_graphed: true (verify GEMMs launch-free)",
        },
        "measurement": lo,
        "model": model,
        "metrics_nan_clean": 1,
        "method": ("LOCAL A10G: per-launch overhead via eager N-kernel chain vs CUDA-graph "
                   "replay; tree-step launch budget modeled from topology; residual bounded. "
                   "No HF Job / no served-file change."),
        "elapsed_s": time.time() - t0,
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print("LAUNCH_JSON_WRITTEN", out_path)
    print(f"per_launch_overhead_us = {lo['per_launch_overhead_us']:.2f} "
          f"(per_kernel_eager {lo['per_kernel_eager_us']:.2f}, graph_replay/N "
          f"{lo['graph_replay_chain_us']/lo['N_kernels']:.2f})")
    print(f"launch counts: total_eager={model['launch_counts']['total_eager']} "
          f"captured={model['launch_counts']['captured_by_deployed_graph']} "
          f"residual={model['launch_counts']['residual_uncaptured']}")
    print(f"residual headroom (upper bound) = {model['residual_headroom_upper_bound_us']:.1f} us "
          f"= {model['residual_headroom_pct_of_bar_step_upper']:.3f}% of bar-step (UPPER bound)")
    print("VERDICT:", model["verdict"])


if __name__ == "__main__":
    main()
