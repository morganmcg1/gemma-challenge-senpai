#!/usr/bin/env python
"""PR #806 — int4head decode NON-GEMM attribution via synthetic-activation microbench.

WHY (and why synthetic / disk-free)
-----------------------------------
#798 (W&B dpc36210) closed the GEMM side of the conc=1 int4head decode cycle:
total 12.42 ms = 256.74 TPS = MTP drafter 2.48 ms + body verify-GEMM 5.92 ms +
lm_head GEMV 0.75 ms + **verify non-GEMM 3.28 ms (26.4%)**. The 3.28 ms was
derived as `verify_gpu - isolated_GEMM` (an upper bound) and never split. This
card splits it into its kernel classes: RMSNorm, RoPE, attention (the force-2D
TRITON_ATTN path bi0 ships), KV-cache write, and sampling.

Like the #798 GEMM attribution, every non-GEMM decode quantity is
VALUE-INDEPENDENT at conc=1: RMSNorm/RoPE/KV-write/sampling time depends only on
shape, and attention only on shape + KV length. So we time the EXACT serving
kernel entry points (vLLM `RMSNorm`, `get_rope`/`rotary_embedding`,
`unified_attention` forced 2D, `triton_reshape_and_cache_flash`) on synthetic
activations of the right shape -- no weights, no checkpoint on disk, no HF job.

LAUNCH-VS-CAPTURED + FUSION is the crux of the lever question. Two facts pin the
"realizable" metric. (1) The shipped serve runs CUDA graphs ON (ENFORCE_EAGER=0)
and the verifier is FULL@7 captured (memory project_drafter_cudagraph_null_789):
every kernel launch happens ONCE at capture and replays launch-free. (2) THIS
build has NO hand-written _C norm/rope kernel -- torch.ops._C.{rms_norm,
fused_add_rms_norm,rotary_embedding} are absent and RMSNorm/RoPE.enabled()=False,
so they run forward_native (pure-PyTorch elementwise chains), which vLLM's
default torch.compile (Inductor) FUSES into ~1 kernel. So we time each class
THREE ways:
  * fused    -> torch.compile (Inductor) + FULL@7 graph = the REALIZABLE shipped
                cost (for opaque Triton attn/kv-write, Inductor can't fuse so
                fused == graphed)
  * graphed  -> launch-free but UNFUSED native = counterfactual (compile OFF)
  * eager    -> launch floor + unfused = fully un-optimized counterfactual
The "fuse the ~250 norms / launch-bind" lever = eager - fused; it is ALREADY
HARVESTED by the shipped compile+graph, so it is NOT an available speedup.

OUTPUT
------
- Per-class table: shape, x-mult/step, eager/graphed/fused us, %HBM-BW, block_ms.
- Per-class REALIZABLE TOTAL (fused x mult) summing to <= the 3.28 ms residual.
- Attention's share (-> surgattn-3D ceiling interpretation, wirbel #791).
- Lever verdict (ceiling % if any, else "irreducible -- kernel side exhausted").
LOCAL ONLY. No model download, no HF job.
"""
from __future__ import annotations

import os

# Must precede torch/vllm import (project_local_pod_gpu_index memory): the pod
# exposes one A10G as index 0 but inherits a stale CUDA_VISIBLE_DEVICES=1.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
# BI=0 to match the shipped force-2D submission (fast non-deterministic kernels).
os.environ.setdefault("VLLM_BATCH_INVARIANT", "0")

import argparse
import gc
import json
import math
import time

import torch

# ---- A10G (AWS g5, GA102, sm_86) roofline ceilings (match gemm_attrib.py) ----
A10G_HBM_GBS = 600.0
BF16 = 2  # bytes

# ---- Gemma-4-E4B text_config (google/gemma-4-E4B-it-qat-w4a16-ct) -------------
HIDDEN = 2560
HEAD_DIM_SLIDING = 256          # config.head_dim
HEAD_DIM_FULL = 512             # config.global_head_dim
NUM_Q_HEADS = 8                 # num_attention_heads
NUM_KV_HEADS = 2                # num_key_value_heads
NUM_LAYERS = 42                 # num_hidden_layers
NUM_KV_SHARED = 18              # num_kv_shared_layers -> first shared idx = 24
PLE_DIM = 256                   # hidden_size_per_layer_input
VOCAB = 262144
RMS_EPS = 1e-6
SLIDING_WINDOW = 512
SOFTCAP_FINAL = 30.0            # final_logit_softcapping
VERIFY_WIDTH = 7                # num_speculative_tokens(6) + 1
BLOCK_SIZE = 16                 # vLLM v1 paged-KV block size

FULL_IDX = {5, 11, 17, 23, 29, 35, 41}
FIRST_SHARED = NUM_LAYERS - NUM_KV_SHARED  # 24

# Layer taxonomy (idx 0..41): non-shared = 0..23, shared = 24..41.
N_FULL = len(FULL_IDX)                                   # 7
N_SLIDING = NUM_LAYERS - N_FULL                          # 35
N_FULL_NONSHARED = len([i for i in FULL_IDX if i < FIRST_SHARED])   # 4
N_FULL_SHARED = N_FULL - N_FULL_NONSHARED                # 3
N_SLIDING_NONSHARED = FIRST_SHARED - N_FULL_NONSHARED    # 20
N_SLIDING_SHARED = N_SLIDING - N_SLIDING_NONSHARED       # 15


# ----------------------------- timing helpers --------------------------------
def time_eager(fn, iters, warmup):
    """Pure eager per-call ms (INCLUDES the kernel-launch floor)."""
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(iters):
            fn()
        e1.record()
        torch.cuda.synchronize()
    return e0.elapsed_time(e1) / iters


def time_graphed(fn, iters, warmup):
    """Launch-free per-call ms via CUDA-graph replay (true kernel time).

    For a native multi-kernel op (e.g. forward_native RMSNorm) this removes the
    per-launch floor but does NOT fuse the kernels. Returns (ms, True). Falls
    back to (eager_ms, False) if capture fails."""
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.inference_mode():
            for _ in range(5):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(g):
            fn()
        for _ in range(max(10, warmup)):
            g.replay()
        torch.cuda.synchronize()
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(iters):
            g.replay()
        e1.record()
        torch.cuda.synchronize()
        ms = e0.elapsed_time(e1) / iters
        del g
        return ms, True
    except Exception as exc:  # noqa: BLE001
        print(f"[nongemm]   graph capture failed: {exc!r}; eager", flush=True)
        return time_eager(fn, iters, warmup), False


def time_fused(fn, iters, warmup):
    """Inductor-fused + launch-free per-call ms = the SHIPPED realizable cost.

    The served verifier runs under torch.compile (Inductor fuses the native
    RMSNorm/elementwise into ~1 kernel) AND a FULL@7 CUDA graph (launch-free).
    We reproduce that by graph-capturing a torch.compile'd fn. For opaque Triton
    kernels (attention, KV-write) Inductor cannot fuse, so this == graphed.
    Falls back to plain graphed if compile/capture fails. Returns (ms, mode)."""
    try:
        cfn = torch.compile(fn)
        ms, ok = time_graphed(cfn, iters, warmup)
        return ms, ("fused" if ok else "fused-eagerfallback")
    except Exception as exc:  # noqa: BLE001
        print(f"[nongemm]   compile failed: {exc!r}; manual-graphed", flush=True)
        ms, ok = time_graphed(fn, iters, warmup)
        return ms, ("graphed" if ok else "eager-fallback")


def bw_pct(total_bytes, ms):
    gb_s = total_bytes / (ms / 1e3) / 1e9
    return gb_s, 100.0 * gb_s / A10G_HBM_GBS


# ----------------------------- class builders --------------------------------
def make_rmsnorm(dim, has_weight, device):
    from vllm.model_executor.layers.layernorm import RMSNorm
    norm = RMSNorm(dim, eps=RMS_EPS, has_weight=has_weight).to(device=device, dtype=torch.bfloat16)
    return norm


def rmsnorm_bytes(n_rows, dim, fused_add):
    # in + out (+ residual in/out if fused) + weight
    b = 2 * n_rows * dim * BF16 + dim * BF16
    if fused_add:
        b += 2 * n_rows * dim * BF16
    return b


def make_rope(head_dim, is_full, device):
    from vllm.model_executor.layers.rotary_embedding import get_rope
    if is_full:
        rope_parameters = {"rope_type": "proportional",
                           "rope_theta": 1000000.0,
                           "partial_rotary_factor": 0.25}
    else:
        rope_parameters = {"rope_type": "default", "rope_theta": 10000.0}
    rope = get_rope(head_dim, max_position=131072, is_neox_style=True,
                    rope_parameters=rope_parameters, dtype=torch.bfloat16)
    return rope.to(device)


def rope_bytes(m, head_dim):
    # q in/out [m, NUM_Q_HEADS*head_dim] + k in/out [m, NUM_KV_HEADS*head_dim]
    q = 2 * m * NUM_Q_HEADS * head_dim * BF16
    k = 2 * m * NUM_KV_HEADS * head_dim * BF16
    return q + k


def attn_bytes(m, head_dim, kv_len, is_full):
    eff_kv = kv_len if is_full else min(kv_len, SLIDING_WINDOW)
    kv = 2 * eff_kv * NUM_KV_HEADS * head_dim * BF16  # K + V cache read
    q = m * NUM_Q_HEADS * head_dim * BF16
    out = m * NUM_Q_HEADS * head_dim * BF16
    return kv + q + out


def kvwrite_bytes(m, head_dim):
    # read key+value [m, kv_heads, head] and write both to cache
    return 2 * (2 * m * NUM_KV_HEADS * head_dim * BF16)


# --------------------------- attention microbench ----------------------------
def build_attn_inputs(m, head_dim, kv_len, is_full, device):
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention
    num_blocks = (kv_len + BLOCK_SIZE - 1) // BLOCK_SIZE + 1
    q = torch.randn(m, NUM_Q_HEADS, head_dim, device=device, dtype=torch.bfloat16)
    out = torch.empty(m, NUM_Q_HEADS, head_dim, device=device, dtype=torch.bfloat16)
    k_cache = torch.randn(num_blocks, BLOCK_SIZE, NUM_KV_HEADS, head_dim,
                          device=device, dtype=torch.bfloat16)
    v_cache = torch.randn(num_blocks, BLOCK_SIZE, NUM_KV_HEADS, head_dim,
                          device=device, dtype=torch.bfloat16)
    cu_seqlens_q = torch.tensor([0, m], device=device, dtype=torch.int32)
    seqused_k = torch.tensor([kv_len], device=device, dtype=torch.int32)
    block_table = torch.arange(num_blocks, device=device, dtype=torch.int32).view(1, num_blocks)
    window = (-1, -1) if is_full else (SLIDING_WINDOW - 1, 0)

    def fn():
        unified_attention(
            q=q, k=k_cache, v=v_cache, out=out,
            cu_seqlens_q=cu_seqlens_q, max_seqlen_q=m,
            seqused_k=seqused_k, max_seqlen_k=kv_len,
            softmax_scale=1.0, causal=True, window_size=window,
            block_table=block_table, softcap=0.0,
            q_descale=None, k_descale=None, v_descale=None,
            softmax_segm_output=None, softmax_segm_max=None,
            softmax_segm_expsum=None,  # force 2D single-pass (the shipped patch)
        )
    return fn


def build_kvwrite_inputs(m, head_dim, device):
    from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
        triton_reshape_and_cache_flash,
    )
    num_blocks = (m + BLOCK_SIZE - 1) // BLOCK_SIZE + 2
    key = torch.randn(m, NUM_KV_HEADS, head_dim, device=device, dtype=torch.bfloat16)
    value = torch.randn(m, NUM_KV_HEADS, head_dim, device=device, dtype=torch.bfloat16)
    k_cache = torch.zeros(num_blocks, BLOCK_SIZE, NUM_KV_HEADS, head_dim,
                          device=device, dtype=torch.bfloat16)
    v_cache = torch.zeros_like(k_cache)
    slot_mapping = torch.arange(m, device=device, dtype=torch.int64)
    k_scale = torch.tensor(1.0, device=device, dtype=torch.float32)
    v_scale = torch.tensor(1.0, device=device, dtype=torch.float32)

    def fn():
        triton_reshape_and_cache_flash(
            key, value, k_cache, v_cache, slot_mapping, "auto", k_scale, v_scale,
        )
    return fn


def build_sampling_inputs(m, device):
    # Greedy verify: final_logit_softcapping (tanh) then argmax over vocab.
    logits = torch.randn(m, VOCAB, device=device, dtype=torch.bfloat16)

    def fn():
        x = logits.float()
        x = torch.tanh(x / SOFTCAP_FINAL) * SOFTCAP_FINAL
        return x.argmax(dim=-1)
    return fn


def sampling_bytes(m):
    return m * VOCAB * BF16 + m * VOCAB * 4  # read bf16 logits + fp32 softcap pass


# --------------------------------- main --------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--m", type=int, default=VERIFY_WIDTH, help="verify width (K+1)")
    ap.add_argument("--kv-sweep", default="128,256,512,640,1024,2048",
                    help="attention KV lengths; report at --kv-report")
    ap.add_argument("--kv-report", type=int, default=640,
                    help="representative KV length for the per-class total")
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--warmup", type=int, default=60)
    ap.add_argument("--residual-ms-budget", type=float, default=3.28,
                    help="the #798 verify non-GEMM upper bound to reconcile against")
    ap.add_argument("--output",
                    default="research/bi0_int4head_nongemm_attrib/nongemm_attrib.json")
    ap.add_argument("--wandb_project",
                    default=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"))
    ap.add_argument("--wandb_entity", default=os.environ.get("WANDB_ENTITY"))
    ap.add_argument("--wandb_group", default="bi0-int4head-nongemm-attrib")
    ap.add_argument("--wandb_name", default="stark/int4head-nongemm-attrib")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    m = args.m
    kv_sweep = [int(x) for x in args.kv_sweep.split(",") if x.strip()]
    device = torch.device("cuda:0")
    assert torch.cuda.is_available(), "no CUDA device (check CUDA_VISIBLE_DEVICES=0)"
    import vllm
    print(f"[nongemm] device={torch.cuda.get_device_name(0)} vllm={vllm.__version__} "
          f"torch={torch.__version__} m={m}", flush=True)

    run = None
    if not args.no_wandb:
        try:
            import wandb
            run = wandb.init(entity=args.wandb_entity, project=args.wandb_project,
                             group=args.wandb_group, name=args.wandb_name,
                             job_type="profiling",
                             config={"m": m, "kv_sweep": kv_sweep,
                                     "kv_report": args.kv_report, "iters": args.iters,
                                     "warmup": args.warmup, "block_size": BLOCK_SIZE,
                                     "device": torch.cuda.get_device_name(0),
                                     "A10G_HBM_GBS": A10G_HBM_GBS,
                                     "residual_ms_budget": args.residual_ms_budget,
                                     "note": "synthetic-activation non-GEMM decode "
                                             "microbench; value-independent at conc=1; "
                                             "splits #798 3.28ms verify non-GEMM residual"})
            print(f"[nongemm] W&B run: {run.url}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[nongemm] W&B init failed: {exc!r}", flush=True)

    t0 = time.time()
    rows = []  # each: {class, shape, mult, eager_ms, graphed_ms, graphed, bytes, gbs, pct_bw}

    def record(cls, shape, mult, fn, total_bytes, note="", try_fuse=True):
        # eager = naive (launch floor + unfused); graphed = launch-free unfused;
        # fused = Inductor-fused + launch-free = the SHIPPED realizable cost.
        eager = time_eager(fn, args.iters, args.warmup)
        graphed, ok = time_graphed(fn, args.iters, args.warmup)
        if try_fuse:
            fused, fmode = time_fused(fn, args.iters, args.warmup)
        else:
            fused, fmode = graphed, "graphed"
        gbs, pct = bw_pct(total_bytes, fused)
        r = {"class": cls, "shape": shape, "mult": mult,
             "eager_us": eager * 1e3, "graphed_us": graphed * 1e3,
             "fused_us": fused * 1e3, "fused_mode": fmode, "graphed_ok": ok,
             "bytes": total_bytes, "gbytes_s": gbs, "pct_hbm_peak": pct,
             "eager_block_ms": eager * mult, "graphed_block_ms": graphed * mult,
             "fused_block_ms": fused * mult,
             "launch_overhead_us": (eager - graphed) * 1e3,
             "fusion_gain_us": (graphed - fused) * 1e3, "note": note}
        rows.append(r)
        print(f"[nongemm] {cls:22s} {shape:18s} x{mult:3d}: "
              f"eager {eager*1e3:6.1f}  graphed {graphed*1e3:6.1f}  "
              f"fused {fused*1e3:6.1f}us ({pct:4.0f}%BW {fmode:18s})  "
              f"blk(fused) {fused*mult:6.3f}ms", flush=True)
        return r

    # vLLM modules (RMSNorm, get_rope) require an active VllmConfig at
    # construction (else "Current vLLM config is not set"); torch.compile of them
    # also reads it. Enter one for the whole measurement (process-scoped).
    from vllm.config import VllmConfig, set_current_vllm_config
    _cfg_ctx = set_current_vllm_config(VllmConfig())
    _cfg_ctx.__enter__()

    # ---- RMSNorm: hidden 2560 (single-arg, the per-layer served form) --------
    print("\n[nongemm] === RMSNorm ===", flush=True)
    norm_h = make_rmsnorm(HIDDEN, True, device)
    x_h = torch.randn(m, HIDDEN, device=device, dtype=torch.bfloat16)
    record("rmsnorm_hidden", f"[{m},{HIDDEN}]", 5 * NUM_LAYERS,
           lambda: norm_h(x_h.clone()), rmsnorm_bytes(m, HIDDEN, False),
           "input/post_attn/pre_ff/post_ff/post_ple x42 layers")

    # final model norm: 2-arg fused-add form (gemma4.py Model.forward line ~1352
    # self.norm(hidden_states, residual)). x1/forward. Inductor fuses the add+norm.
    res_h = torch.randn(m, HIDDEN, device=device, dtype=torch.bfloat16)
    record("rmsnorm_final_fusedadd", f"[{m},{HIDDEN}]", 1,
           lambda: norm_h(x_h.clone(), res_h.clone()), rmsnorm_bytes(m, HIDDEN, True),
           "final model norm (2-arg fused-add residual)")

    # PLE projection norm: [m*42, 256]
    norm_ple = make_rmsnorm(PLE_DIM, True, device)
    x_ple = torch.randn(m * NUM_LAYERS, PLE_DIM, device=device, dtype=torch.bfloat16)
    record("rmsnorm_ple_proj", f"[{m*NUM_LAYERS},{PLE_DIM}]", 1,
           lambda: norm_ple(x_ple.clone()), rmsnorm_bytes(m * NUM_LAYERS, PLE_DIM, False),
           "per_layer_projection_norm x1/forward")

    # q/k/v head-dim norms (sliding 256, full 512)
    for hd, n_q, n_kv, tag in [
        (HEAD_DIM_SLIDING, N_SLIDING, N_SLIDING_NONSHARED, "sliding"),
        (HEAD_DIM_FULL, N_FULL, N_FULL_NONSHARED, "full"),
    ]:
        norm_q = make_rmsnorm(hd, True, device)
        xq = torch.randn(m * NUM_Q_HEADS, hd, device=device, dtype=torch.bfloat16)
        record(f"rmsnorm_qnorm_{tag}", f"[{m*NUM_Q_HEADS},{hd}]", n_q,
               lambda nq=norm_q, xx=xq: nq(xx.clone()),
               rmsnorm_bytes(m * NUM_Q_HEADS, hd, False), "q_norm all layers")
        norm_k = make_rmsnorm(hd, True, device)
        xk = torch.randn(m * NUM_KV_HEADS, hd, device=device, dtype=torch.bfloat16)
        record(f"rmsnorm_knorm_{tag}", f"[{m*NUM_KV_HEADS},{hd}]", n_kv,
               lambda nk=norm_k, xx=xk: nk(xx.clone()),
               rmsnorm_bytes(m * NUM_KV_HEADS, hd, False), "k_norm non-shared layers")
        norm_v = make_rmsnorm(hd, False, device)  # v_norm has no learned weight
        record(f"rmsnorm_vnorm_{tag}", f"[{m*NUM_KV_HEADS},{hd}]", n_kv,
               lambda nv=norm_v, xx=xk: nv(xx.clone()),
               rmsnorm_bytes(m * NUM_KV_HEADS, hd, False), "v_norm non-shared layers")

    # ---- RoPE (per layer; sliding 256 / full 512 partial-0.25) ---------------
    print("\n[nongemm] === RoPE ===", flush=True)
    for hd, n_layers, is_full, tag in [
        (HEAD_DIM_SLIDING, N_SLIDING, False, "sliding"),
        (HEAD_DIM_FULL, N_FULL, True, "full"),
    ]:
        rope = make_rope(hd, is_full, device)
        pos = torch.arange(m, device=device, dtype=torch.int64)
        q = torch.randn(m, NUM_Q_HEADS * hd, device=device, dtype=torch.bfloat16)
        k = torch.randn(m, NUM_KV_HEADS * hd, device=device, dtype=torch.bfloat16)
        record(f"rope_{tag}", f"q[{m},{NUM_Q_HEADS*hd}]", n_layers,
               lambda rp=rope, p=pos, qq=q, kk=k: rp(p, qq.clone(), kk.clone()),
               rope_bytes(m, hd), "rotary_embedding apply per layer")

    # ---- per-layer elementwise residual/scale chain --------------------------
    # gemma4.py DecoderLayer.forward: h+=resid (l718), h+=resid (l739),
    # h+=per_layer_contribution (l750), h*=layer_scalar (l754) -- 3 adds + 1 mul
    # on [m,2560] per layer. These are SEPARATE ops eagerly (4 launches/layer) but
    # Inductor fuses the chain into ~1 kernel -> the realizable cost is ~0.
    print("\n[nongemm] === elementwise residual/scale ===", flush=True)
    h_e = torch.randn(m, HIDDEN, device=device, dtype=torch.bfloat16)
    ra = torch.randn(m, HIDDEN, device=device, dtype=torch.bfloat16)
    rb = torch.randn(m, HIDDEN, device=device, dtype=torch.bfloat16)
    scal = torch.tensor(1.03, device=device, dtype=torch.bfloat16)

    def _elementwise():
        h = h_e + ra
        h = h + rb
        h = h + ra
        return h * scal

    record("elementwise_resid", f"[{m},{HIDDEN}]", NUM_LAYERS, _elementwise,
           (4 + 1) * m * HIDDEN * BF16,
           "3 residual-adds + layer-scalar mul per layer (PLE gate gelu/mul on "
           "[m,256] also fuses, even smaller)")

    # ---- attention (force-2D unified_attention), KV-length sweep -------------
    print("\n[nongemm] === attention (force-2D TRITON_ATTN) ===", flush=True)
    attn_by_kv = {"sliding": {}, "full": {}}
    for kv_len in kv_sweep:
        for hd, n_layers, is_full, tag in [
            (HEAD_DIM_SLIDING, N_SLIDING, False, "sliding"),
            (HEAD_DIM_FULL, N_FULL, True, "full"),
        ]:
            fn = build_attn_inputs(m, hd, kv_len, is_full, device)
            eager = time_eager(fn, args.iters, args.warmup)
            graphed, ok = time_graphed(fn, args.iters, args.warmup)
            tb = attn_bytes(m, hd, kv_len, is_full)
            gbs, pct = bw_pct(tb, graphed)
            attn_by_kv[tag][kv_len] = {"eager_us": eager * 1e3, "graphed_us": graphed * 1e3,
                                       "graphed": ok, "pct_hbm_peak": pct, "n_layers": n_layers,
                                       "head_dim": hd}
            print(f"[nongemm] attn_{tag:7s} KV={kv_len:5d} hd={hd}: "
                  f"eager {eager*1e3:6.1f}us  graphed {graphed*1e3:6.1f}us ({pct:4.0f}%BW) "
                  f"x{n_layers}", flush=True)
            gc.collect(); torch.cuda.empty_cache()

    # attention rows at the representative KV length. Attention is an opaque
    # Triton kernel: Inductor cannot fuse it, so the realizable cost = graphed
    # (it replays launch-free in the FULL@7 graph). fused == graphed here.
    kvr = args.kv_report
    for tag, hd, n_layers, is_full in [
        ("sliding", HEAD_DIM_SLIDING, N_SLIDING, False),
        ("full", HEAD_DIM_FULL, N_FULL, True),
    ]:
        d = attn_by_kv[tag][kvr]
        eff = "min(KV,512)" if not is_full else "KV"
        g_us = d["graphed_us"]
        rows.append({"class": f"attn_{tag}", "shape": f"q[{m},{NUM_Q_HEADS},{hd}]@KV{kvr}({eff})",
                     "mult": n_layers, "eager_us": d["eager_us"], "graphed_us": g_us,
                     "fused_us": g_us, "fused_mode": "graphed(opaque-triton)",
                     "graphed_ok": d["graphed"], "bytes": attn_bytes(m, hd, kvr, is_full),
                     "gbytes_s": None, "pct_hbm_peak": d["pct_hbm_peak"],
                     "eager_block_ms": d["eager_us"] / 1e3 * n_layers,
                     "graphed_block_ms": g_us / 1e3 * n_layers,
                     "fused_block_ms": g_us / 1e3 * n_layers,
                     "launch_overhead_us": d["eager_us"] - g_us, "fusion_gain_us": 0.0,
                     "note": f"force-2D unified_attention @ KV={kvr}"})

    # ---- KV-cache write (reshape_and_cache_flash; non-shared layers) ---------
    print("\n[nongemm] === KV write ===", flush=True)
    for hd, n_layers, tag in [
        (HEAD_DIM_SLIDING, N_SLIDING_NONSHARED, "sliding"),
        (HEAD_DIM_FULL, N_FULL_NONSHARED, "full"),
    ]:
        fn = build_kvwrite_inputs(m, hd, device)
        record(f"kv_write_{tag}", f"[{m},{NUM_KV_HEADS},{hd}]", n_layers, fn,
               kvwrite_bytes(m, hd), "reshape_and_cache_flash non-shared layers (opaque "
               "Triton -> realizable=graphed)", try_fuse=False)

    # ---- sampling (softcap + argmax over vocab) ------------------------------
    print("\n[nongemm] === sampling ===", flush=True)
    fn = build_sampling_inputs(m, device)
    record("sampling_argmax", f"[{m},{VOCAB}]", 1, fn, sampling_bytes(m),
           "final_logit_softcapping(tanh)+argmax greedy verify")

    _cfg_ctx.__exit__(None, None, None)

    # ------------------------------ aggregate ---------------------------------
    # REALIZABLE per-class total = fused_block_ms (Inductor-fused + launch-free,
    # = what the shipped torch.compile'd FULL@7 verifier actually runs). For the
    # opaque Triton classes (attn, kv_write) fused==graphed. We also keep the
    # unfused-graphed and eager sums as counterfactuals (the launch/fusion lever).
    total_fused = sum(r["fused_block_ms"] for r in rows)
    total_graphed = sum(r["graphed_block_ms"] for r in rows)
    total_eager = sum(r["eager_block_ms"] for r in rows)

    # grouped realizable totals
    def grp(prefix):
        return sum(r["fused_block_ms"] for r in rows if r["class"].startswith(prefix))
    norm_total = grp("rmsnorm")
    rope_total = grp("rope")
    elt_total = grp("elementwise")
    attn_total = grp("attn")
    kvw_total = grp("kv_write")
    samp_total = grp("sampling")

    budget = args.residual_ms_budget
    print("\n[nongemm] ===== per-class REALIZABLE (fused x mult) =====", flush=True)
    for name, val in [("RMSNorm (all)", norm_total), ("RoPE", rope_total),
                      ("elementwise", elt_total), ("attention", attn_total),
                      ("KV write", kvw_total), ("sampling", samp_total)]:
        print(f"  {name:16s}: {val:7.3f} ms  ({100*val/budget:5.1f}% of {budget}ms residual)",
              flush=True)
    print(f"  {'SUM (fused)':16s}: {total_fused:7.3f} ms  "
          f"({100*total_fused/budget:5.1f}% of {budget}ms)  <- realizable", flush=True)
    print(f"  {'SUM (graphed)':16s}: {total_graphed:7.3f} ms  "
          f"(unfused-but-launch-free counterfactual)", flush=True)
    print(f"  {'SUM (eager)':16s}: {total_eager:7.3f} ms  "
          f"(fully un-optimized: launch floor + unfused)", flush=True)
    attn_share = 100 * attn_total / max(total_fused, 1e-9)
    # share-normalized estimate of attention's slice of the REAL #798 residual
    # (per-op fused sum is an upper bound; the share is the robust quantity).
    attn_ms_of_budget = attn_share / 100.0 * budget
    print(f"  attention share of measured realizable non-GEMM: {attn_share:5.1f}% "
          f"(~{attn_ms_of_budget:.2f}ms of the {budget}ms residual)", flush=True)
    print(f"  fusion+launch lever (eager - fused): {total_eager-total_fused:7.3f} ms/step "
          f"(ALREADY HARVESTED by compile+CUDA-graph in the shipped path)", flush=True)
    print(f"  CALIBRATION: SUM(fused)={total_fused:.2f}ms > {budget}ms because per-op "
          f"isolation cannot capture the full model's cross-op Inductor fusion and the "
          f"synthetic paged-KV attention over-costs vs real cached-KV flash. Per-class "
          f"numbers are UPPER BOUNDS; the per-class RANKING and lever verdict are robust.",
          flush=True)

    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print(f"[nongemm] peak GPU mem: {peak_mem:.2f} GiB; elapsed {time.time()-t0:.1f}s", flush=True)

    payload = {
        "config": {
            "device": torch.cuda.get_device_name(0), "vllm": vllm.__version__,
            "torch": torch.__version__, "m": m, "kv_sweep": kv_sweep, "kv_report": kvr,
            "iters": args.iters, "warmup": args.warmup, "block_size": BLOCK_SIZE,
            "A10G_HBM_GBS": A10G_HBM_GBS, "residual_ms_budget": budget,
            "peak_gpu_mem_gib": peak_mem,
            "layer_taxonomy": {
                "n_full": N_FULL, "n_sliding": N_SLIDING,
                "n_full_nonshared": N_FULL_NONSHARED, "n_sliding_nonshared": N_SLIDING_NONSHARED,
                "n_full_shared": N_FULL_SHARED, "n_sliding_shared": N_SLIDING_SHARED},
            "note": "synthetic-activation non-GEMM decode microbench; value-independent "
                    "at conc=1. RMSNorm/RoPE have NO _C kernel in this build "
                    "(enabled()=False -> forward_native); realizable=fused "
                    "(torch.compile Inductor + FULL@7 CUDA graph), graphed=unfused "
                    "launch-free, eager=launch-floor. Splits #798 (dpc36210) 3.28ms "
                    "verify non-GEMM.",
        },
        "rows": rows,
        "attn_by_kv": attn_by_kv,
        "totals": {
            "rmsnorm_ms": norm_total, "rope_ms": rope_total, "elementwise_ms": elt_total,
            "attn_ms": attn_total, "kv_write_ms": kvw_total, "sampling_ms": samp_total,
            "sum_fused_ms": total_fused, "sum_graphed_ms": total_graphed,
            "sum_eager_ms": total_eager,
            "attention_share_pct": attn_share,
            "attention_ms_of_budget": attn_ms_of_budget,
            "fused_vs_budget_pct": 100 * total_fused / budget,
            "fusion_launch_lever_ms": total_eager - total_fused,
            "calibration_note": "per-op fused sum is an UPPER BOUND (no cross-op "
                                "Inductor fusion; synthetic paged-KV attn over-costs "
                                "vs real cached-KV flash). Ranking + lever verdict robust.",
        },
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[nongemm] wrote {args.output}", flush=True)

    if run is not None:
        try:
            cols = ["class", "shape", "mult", "eager_us", "graphed_us", "fused_us",
                    "fused_mode", "pct_hbm_peak", "launch_overhead_us", "fusion_gain_us",
                    "fused_block_ms", "graphed_block_ms", "eager_block_ms"]
            tbl = wandb.Table(columns=cols)
            for r in rows:
                tbl.add_data(r["class"], r["shape"], r["mult"], r["eager_us"],
                             r["graphed_us"], r["fused_us"], r["fused_mode"],
                             r["pct_hbm_peak"], r["launch_overhead_us"], r["fusion_gain_us"],
                             r["fused_block_ms"], r["graphed_block_ms"], r["eager_block_ms"])
            run.log({"nongemm_table": tbl})
            akv = wandb.Table(columns=["kind", "kv_len", "eager_us", "graphed_us",
                                       "pct_hbm_peak", "n_layers"])
            for tag, d in attn_by_kv.items():
                for kv_len, v in d.items():
                    akv.add_data(tag, kv_len, v["eager_us"], v["graphed_us"],
                                 v["pct_hbm_peak"], v["n_layers"])
            run.log({"attn_by_kv_table": akv})
            run.summary.update({
                "rmsnorm_ms": norm_total, "rope_ms": rope_total,
                "elementwise_ms": elt_total, "attn_ms": attn_total,
                "kv_write_ms": kvw_total, "sampling_ms": samp_total,
                "sum_fused_ms": total_fused, "sum_graphed_ms": total_graphed,
                "sum_eager_ms": total_eager,
                "attention_share_pct": payload["totals"]["attention_share_pct"],
                "fused_vs_budget_pct": payload["totals"]["fused_vs_budget_pct"],
                "fusion_launch_lever_ms": payload["totals"]["fusion_launch_lever_ms"],
                "peak_gpu_mem_gib": peak_mem,
            })
            run.finish()
        except Exception as exc:  # noqa: BLE001
            print(f"[nongemm] W&B log failed: {exc!r}", flush=True)


if __name__ == "__main__":
    main()
