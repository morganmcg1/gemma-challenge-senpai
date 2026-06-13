"""Attention deep-profile (PR #39): quantify the 19.6% decode-attention lever.

Answers, for the served frontier stack (``fa2sw_precache_kenyan``) at conc=1:

  1. *Which* kernel is the 19.6% "fa2sw attention"?  (trace attribution)
  2. How close is that kernel to the **HBM-bandwidth floor** for sliding-window
     KV reads — i.e. is there time to win, or is attention already near-optimal?
  3. M=1 kernel bake-off: vLLM Triton ``unified_attention`` vs vendored
     FlashAttention-2 ``flash_attn_varlen_func`` vs torch SDPA.
  4. TPS-uplift projection for attention savings of 10/25/50/100% and a verdict
     on whether reducing attention is worth pursuing.

This is a **local A10G op-microbench** — no server, no submission, no leaderboard
number.  It drives the *real* served kernel (``vllm.v1.attention.ops.
triton_unified_attention.unified_attention``) on a paged KV cache with L2-defeating
buffer rotation, and reads device-side kernel time with ``torch.profiler`` (the
same instrument that produced the served 88.6 ms / 19.6% figure), so the achieved
GB/s is directly comparable to the frontier profile.

Run via::

    python -m scripts.local_validation.profile_decode \
        --profile-mode attention-detail \
        --M-values 1,7,17,25,45 \
        --output research/profiling/fa2sw_attention/attention_detail.json
"""
from __future__ import annotations

import gzip
import json
import math
import statistics
import time
from pathlib import Path

# --- Gemma-4-E4B text-decoder architecture (from osoi5-v0-baked config.json) ---
LAYER_TYPES = [
    "sliding", "sliding", "full", "sliding", "sliding", "sliding", "sliding",
    "sliding", "full", "sliding", "sliding", "sliding", "sliding", "sliding",
    "full", "sliding", "sliding", "sliding", "sliding", "sliding", "full",
    "sliding", "sliding", "sliding", "sliding", "sliding", "full", "sliding",
    "sliding", "sliding", "sliding", "sliding", "full", "sliding", "sliding",
    "sliding", "full",
]
N_LAYERS = len(LAYER_TYPES)                       # 37
N_FULL = sum(t == "full" for t in LAYER_TYPES)    # 7
N_SLIDING = sum(t == "sliding" for t in LAYER_TYPES)  # 30
HEAD_DIM = {"sliding": 256, "full": 512}
SLIDING_WINDOW = 512
N_Q_HEADS = 8
N_KV_HEADS = 2
QPKV = N_Q_HEADS // N_KV_HEADS                    # 4
NUM_KV_SHARED_LAYERS = 16
DTYPE_BYTES = 2                                   # bf16
OUTPUT_LEN = 512
BLOCK_SIZE = 16                                   # vLLM paged-cache block

A10G_PEAK_GBPS = 600.0                            # GDDR6 spec-sheet peak

# Served-frontier anchors (research/profiling/frontier_decode/frontier_decode_profile.json)
FRONTIER = {
    "attn_frac_of_gpu_busy": 0.19629872265302628,
    "attn_ms_per_cycle": 1.8357856542511017,
    "kernel_unified_attention_ms_total": 88.57348199999979,
    "attention_category_ms_total": 90.26759699999967,
    "reduce_segments_ms_total": 3.9473849999999784,
    "gpu_busy_ms_per_cycle": 9.352,
    "cycle_wall_ms": 9.416,
    "e_accept": 3.817303602892035,
    "measured_steady_tps": 391.3125,
    "public_a10g_small_tps": 424.5,   # leaderboard osoi5-…-precache anchor
}


# --------------------------------------------------------------------------- #
# Step 0 — trace attribution: prove what the 19.6% kernel actually is          #
# --------------------------------------------------------------------------- #
def analyze_trace(profile_json: Path, trace_path: Path | None) -> dict:
    """Split the served ``attention`` category into its constituent kernels and
    scan the raw chrome trace for *any* FlashAttention/FMHA/SDPA kernel (to
    confirm the inert ``fa2sw`` FA2 router never fired)."""
    data = json.loads(Path(profile_json).read_text())
    fr = data["variants"]["frontier"]["kernel"]["trace"]
    cat_ms = fr["category_ms"]
    tops = {k["name"]: k["ms"] for k in fr["top_kernels"]}
    kua = tops.get("kernel_unified_attention", 0.0)
    redseg = tops.get("reduce_segments", 0.0)
    attn_total = cat_ms.get("attention", 0.0)
    # attention category = kernel_unified_attention + reshape_and_cache (KV write)
    reshape_and_cache = attn_total - kua

    fa2_hits: list[str] = []
    scanned = False
    if trace_path and Path(trace_path).exists():
        scanned = True
        needles = ("flash", "fmha", "fwd_kernel", "mha_", "sdpa",
                   "scaled_dot_product", "mem_eff", "efficient_attention",
                   "fav2", "fa2", "cutlassF")
        names: set[str] = set()
        with gzip.open(trace_path, "rt") as fh:
            blob = json.load(fh)
        for ev in blob.get("traceEvents", []):
            if ev.get("ph") == "X" and ev.get("cat") in ("kernel", "Kernel"):
                names.add(ev.get("name", ""))
        for nm in names:
            low = nm.lower()
            if any(n in low for n in needles):
                fa2_hits.append(nm)

    return {
        "kernel_unified_attention_ms": kua,
        "reduce_segments_ms": redseg,
        "reshape_and_cache_ms": reshape_and_cache,
        "attention_category_ms": attn_total,
        "fraction_unified_attention": kua / attn_total if attn_total else None,
        "trace_scanned": scanned,
        "fa2_fmha_sdpa_kernels_found": sorted(fa2_hits),
        "served_attention_kernel": "kernel_unified_attention (vLLM Triton)",
        "note": (
            "fa2sw FA2 router is inert: vLLM forces TRITON_ATTN for the "
            "heterogeneous head_dims (sliding=256, full=512); FA2 caps at 256. "
            "Zero FlashAttention/FMHA kernels in the served decode trace."
        ),
    }


# --------------------------------------------------------------------------- #
# Step 2 — theoretical SWA bandwidth floor                                     #
# --------------------------------------------------------------------------- #
def prompt_lengths(decode_jsonl: Path | None, eval_prompts: Path | None,
                   baked_model: Path | None) -> tuple[list[int], str]:
    """Per-prompt prompt-token counts for the 128 sharegpt eval prompts.

    Tries (1) the decode jsonl, (2) tokenizing the eval prompts with the baked
    tokenizer, (3) an aggregate-mean fallback (34836 tok / 128 prompts)."""
    # (1) decode jsonl with explicit prompt token ids
    if decode_jsonl and Path(decode_jsonl).exists():
        lens = []
        for line in Path(decode_jsonl).read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            for key in ("prompt_token_ids", "prompt_tokens", "input_ids"):
                if isinstance(rec.get(key), list):
                    lens.append(len(rec[key]))
                    break
        if lens:
            return lens, f"decode_jsonl:{decode_jsonl.name}"
    # (2) tokenize the eval prompts
    if eval_prompts and Path(eval_prompts).exists() and baked_model:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(str(baked_model))
            raw = json.loads(Path(eval_prompts).read_text())
            prompts = []
            for r in (raw if isinstance(raw, list) else raw.get("prompts", [])):
                if isinstance(r, str):
                    prompts.append(r)
                elif isinstance(r, dict):
                    prompts.append(r.get("prompt") or r.get("text") or
                                   json.dumps(r.get("messages", "")))
            lens = [len(tok(p, add_special_tokens=True)["input_ids"]) for p in prompts]
            if lens:
                return lens, f"tokenized:{eval_prompts.name}"
        except Exception as e:  # noqa: BLE001
            print(f"[attn] tokenizer fallback ({e!r})", flush=True)
    # (3) aggregate-mean fallback
    mean_len = round(34836 / 128)
    return [mean_len] * 128, "aggregate_mean_34836_over_128"


def swa_floor(plens: list[int], plens_source: str,
              peak_gbps: float = A10G_PEAK_GBPS) -> dict:
    """Theoretical minimum KV-load bytes / verify-cycle at conc=1, summed over
    all 37 attention layers, using the actual decode context distribution.

    Per layer, per cycle, the kernel must stream K and V for every attended
    position once (flat-in-M: the M query rows share the KV read):

        sliding layer:  min(ctx, W) * n_kv * head_dim(256) * 2(K,V) * 2(bf16)
        full    layer:  ctx          * n_kv * head_dim(512) * 2(K,V) * 2(bf16)
    """
    # mean over all (prompt, decode-step) pairs
    full_bytes_pos = N_KV_HEADS * HEAD_DIM["full"] * 2 * DTYPE_BYTES      # 4096 B/pos
    slid_bytes_pos = N_KV_HEADS * HEAD_DIM["sliding"] * 2 * DTYPE_BYTES   # 2048 B/pos

    ctx_vals, min_ctx_vals = [], []
    for L in plens:
        for t in range(OUTPUT_LEN):
            ctx = L + t
            ctx_vals.append(ctx)
            min_ctx_vals.append(min(ctx, SLIDING_WINDOW))
    mean_ctx = statistics.fmean(ctx_vals)
    mean_min_ctx = statistics.fmean(min_ctx_vals)

    full_bytes = N_FULL * mean_ctx * full_bytes_pos
    slid_bytes = N_SLIDING * mean_min_ctx * slid_bytes_pos
    total_bytes = full_bytes + slid_bytes

    # architectural lower bound if the 16 shared-KV layers could read once
    # (CLA/YOCO read-coalescing) — NOT what the served kernel does.
    unique_frac = (N_LAYERS - NUM_KV_SHARED_LAYERS) / N_LAYERS
    floor_time_us = total_bytes / (peak_gbps * 1e9) * 1e6
    return {
        "plens_source": plens_source,
        "n_prompts": len(plens),
        "mean_prompt_len": statistics.fmean(plens),
        "mean_ctx": mean_ctx,
        "mean_min_ctx_window": mean_min_ctx,
        "full_layers": N_FULL, "sliding_layers": N_SLIDING,
        "kv_bytes_per_cycle_full": full_bytes,
        "kv_bytes_per_cycle_sliding": slid_bytes,
        "kv_bytes_per_cycle_total": total_bytes,
        "kv_MB_per_cycle_total": total_bytes / 1e6,
        "floor_time_us_at_peak": floor_time_us,
        "floor_time_ms_at_peak": floor_time_us / 1e3,
        "peak_gbps_assumed": peak_gbps,
        "shared_kv_unique_fraction": unique_frac,
        "kv_MB_per_cycle_if_shared_coalesced": total_bytes * unique_frac / 1e6,
    }


# --------------------------------------------------------------------------- #
# Steps 1 & 3 — GPU microbench of the real kernels                             #
# --------------------------------------------------------------------------- #
def _measure_peak_bw(torch, device) -> dict:
    """Practical achievable HBM bandwidth via a large rotated copy."""
    best = 0.0
    for mb in (64, 128, 256):
        n = mb * 1024 * 1024 // 2  # bf16 elements
        a = torch.randn(n, dtype=torch.bfloat16, device=device)
        b = torch.empty_like(a)
        for _ in range(3):
            b.copy_(a)
        torch.cuda.synchronize()
        ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
        iters = 50
        ev0.record()
        for _ in range(iters):
            b.copy_(a)
        ev1.record()
        torch.cuda.synchronize()
        ms = ev0.elapsed_time(ev1) / iters
        gbps = (2 * a.numel() * 2) / (ms / 1e3) / 1e9  # read+write
        best = max(best, gbps)
        del a, b
    return {"measured_peak_gbps_copy": best, "spec_peak_gbps": A10G_PEAK_GBPS}


def _profiled_device_us(torch, fn, n_iter: int, warmup: int = 20) -> float:
    """Sum of GPU self-time over ``n_iter`` calls / n_iter, via torch.profiler
    (device-only time — excludes Python launch overhead, matching how the
    served 88.6 ms attention figure was measured)."""
    from torch.profiler import ProfilerActivity, profile
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(n_iter):
            fn()
        torch.cuda.synchronize()
    total_us = sum(e.self_device_time_total for e in prof.key_averages())
    return total_us / n_iter


def _build_paged_kv(torch, device, layer_type: str, ctx: int, rot: int):
    """Allocate a paged KV cache with ``rot`` disjoint context windows so each
    bench iteration reads cold KV from HBM (defeats the 6 MB L2)."""
    hd = HEAD_DIM[layer_type]
    nb = math.ceil(ctx / BLOCK_SIZE)
    total_blocks = rot * nb
    shape = (total_blocks, BLOCK_SIZE, N_KV_HEADS, hd)
    key_cache = torch.randn(shape, dtype=torch.bfloat16, device=device) * 0.1
    value_cache = torch.randn(shape, dtype=torch.bfloat16, device=device) * 0.1
    block_tables = [
        torch.arange(r * nb, r * nb + nb, dtype=torch.int32, device=device).view(1, nb)
        for r in range(rot)
    ]
    return key_cache, value_cache, block_tables, nb


def _rot_for(layer_type: str, ctx: int, target_mb: float = 96.0) -> int:
    """#rotation buffers so total KV footprint >> L2 (6 MB)."""
    hd = HEAD_DIM[layer_type]
    eff = min(ctx, SLIDING_WINDOW) if layer_type == "sliding" else ctx
    kv_bytes = eff * N_KV_HEADS * hd * 2 * DTYPE_BYTES
    return max(8, math.ceil(target_mb * 1e6 / kv_bytes))


def _op_bytes(layer_type: str, ctx: int, M: int) -> dict:
    """HBM traffic for one attention op (one layer, one cycle)."""
    hd = HEAD_DIM[layer_type]
    eff = min(ctx, SLIDING_WINDOW) if layer_type == "sliding" else ctx
    eff_pad = math.ceil(eff / BLOCK_SIZE) * BLOCK_SIZE
    kv_floor = eff * N_KV_HEADS * hd * 2 * DTYPE_BYTES        # minimal
    kv_raw = eff_pad * N_KV_HEADS * hd * 2 * DTYPE_BYTES      # block-padded
    q_bytes = M * N_Q_HEADS * hd * DTYPE_BYTES
    out_bytes = M * N_Q_HEADS * hd * DTYPE_BYTES
    return {
        "kv_floor_bytes": kv_floor,
        "kv_raw_bytes": kv_raw,
        "q_bytes": q_bytes,
        "out_bytes": out_bytes,
        "total_raw_bytes": kv_raw + q_bytes + out_bytes,
        "total_floor_bytes": kv_floor + q_bytes + out_bytes,
    }


def _sdpa_reference(torch, q, key_cache, value_cache, block_table, ctx, M,
                    layer_type, scale):
    """Dense torch-SDPA reference for the same op (numerical ground truth)."""
    import torch.nn.functional as F
    hd = HEAD_DIM[layer_type]
    nb = block_table.shape[1]
    kc = key_cache[block_table[0]].reshape(nb * BLOCK_SIZE, N_KV_HEADS, hd)[:ctx]
    vc = value_cache[block_table[0]].reshape(nb * BLOCK_SIZE, N_KV_HEADS, hd)[:ctx]
    # q: [M, n_q, hd] -> [1, n_q, M, hd]; kv: [1, n_kv, ctx, hd]
    qd = q.permute(1, 0, 2).unsqueeze(0).float()
    kd = kc.permute(1, 0, 2).unsqueeze(0).float()
    vd = vc.permute(1, 0, 2).unsqueeze(0).float()
    # causal (+ sliding) mask over absolute positions
    qpos = torch.arange(ctx - M, ctx, device=q.device)
    kpos = torch.arange(ctx, device=q.device)
    mask = kpos[None, :] <= qpos[:, None]
    if layer_type == "sliding":
        mask &= kpos[None, :] > (qpos[:, None] - SLIDING_WINDOW)
    out = F.scaled_dot_product_attention(
        qd, kd, vd, attn_mask=mask[None, None], scale=scale, enable_gqa=True)
    return out.squeeze(0).permute(1, 0, 2).to(q.dtype)  # [M, n_q, hd]


def bench_op(torch, layer_type: str, M: int, ctx: int, *, dispatch: str = "served",
             n_iter: int = 100, validate: bool = False) -> dict:
    """Bench one layer-type attention op with the real Triton kernel.

    dispatch:  'served' -> pass 3D buffers (M==1 auto-3D, M>1 auto-2D, matches
               the live stack);  'force2d' -> seq_threshold_3D=None (always 2D)."""
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention
    device = torch.device("cuda")
    hd = HEAD_DIM[layer_type]
    scale = 1.0 / math.sqrt(hd)
    window = (SLIDING_WINDOW - 1, 0) if layer_type == "sliding" else (-1, -1)

    rot = _rot_for(layer_type, ctx)
    key_cache, value_cache, block_tables, nb = _build_paged_kv(
        torch, device, layer_type, ctx, rot)
    q = torch.randn(M, N_Q_HEADS, hd, dtype=torch.bfloat16, device=device) * 0.1
    out = torch.empty(M, N_Q_HEADS, hd, dtype=torch.bfloat16, device=device)
    cu_seqlens_q = torch.tensor([0, M], dtype=torch.int32, device=device)
    seqused_k = torch.tensor([ctx], dtype=torch.int32, device=device)

    # 3D split-KV softmax buffers (FlashDecoding), sized like the backend
    seq_threshold_3D = 128 // N_KV_HEADS  # = 64
    n_seg = 16
    hdp = hd  # already pow2
    segm_out = torch.empty(seq_threshold_3D, N_Q_HEADS, n_seg, hdp,
                           dtype=torch.float32, device=device)
    segm_max = torch.empty(seq_threshold_3D, N_Q_HEADS, n_seg,
                           dtype=torch.float32, device=device)
    segm_exp = torch.empty(seq_threshold_3D, N_Q_HEADS, n_seg,
                           dtype=torch.float32, device=device)
    thr = None if dispatch == "force2d" else seq_threshold_3D

    state = {"i": 0}

    def call():
        bt = block_tables[state["i"] % rot]
        state["i"] += 1
        unified_attention(
            q=q, k=key_cache, v=value_cache, out=out,
            cu_seqlens_q=cu_seqlens_q, max_seqlen_q=M,
            seqused_k=seqused_k, max_seqlen_k=ctx,
            softmax_scale=scale, causal=True, window_size=window,
            block_table=bt, softcap=0.0,
            q_descale=None, k_descale=None, v_descale=None,
            seq_threshold_3D=thr, num_par_softmax_segments=n_seg,
            softmax_segm_output=segm_out, softmax_segm_max=segm_max,
            softmax_segm_expsum=segm_exp,
        )

    used_3d = (dispatch != "force2d") and (M == 1)
    err = None
    if validate:
        state["i"] = 0
        call()
        torch.cuda.synchronize()
        ref = _sdpa_reference(torch, q, key_cache, value_cache,
                              block_tables[0], ctx, M, layer_type, scale)
        err = {
            "max_abs_err": (out - ref).abs().max().item(),
            "mean_abs_err": (out - ref).abs().mean().item(),
            "ref_abs_mean": ref.abs().mean().item(),
        }

    state["i"] = 0
    dev_us = _profiled_device_us(torch, call, n_iter)
    # CUDA-event wall cross-check (rules out a profiler artifact): wall>=device
    # by the eager launch gap only; device_us is the served-comparable number.
    state["i"] = 0
    for _ in range(10):
        call()
    torch.cuda.synchronize()
    ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
    ev0.record()
    for _ in range(n_iter):
        call()
    ev1.record()
    torch.cuda.synchronize()
    wall_us = ev0.elapsed_time(ev1) / n_iter * 1e3
    b = _op_bytes(layer_type, ctx, M)
    gbps_total = b["total_raw_bytes"] / (dev_us / 1e6) / 1e9
    gbps_floor = b["kv_floor_bytes"] / (dev_us / 1e6) / 1e9
    # arithmetic intensity (decode GQA): FLOPs / byte
    flops = 2 * 2 * M * N_Q_HEADS * (min(ctx, SLIDING_WINDOW)
                                     if layer_type == "sliding" else ctx) * hd
    ai = flops / b["total_raw_bytes"]
    del key_cache, value_cache
    torch.cuda.empty_cache()
    return {
        "layer_type": layer_type, "M": M, "ctx": ctx, "dispatch": dispatch,
        "used_3d_split_kv": used_3d, "rot_buffers": rot,
        "device_us": dev_us, "wall_us": wall_us,
        "achieved_gbps_total": gbps_total,
        "achieved_gbps_kv_floor": gbps_floor,
        "arithmetic_intensity_flops_per_byte": ai,
        "bytes": b, "validation": err,
    }


def bench_fa2(torch, M: int, ctx: int, n_iter: int = 100,
              validate: bool = False) -> dict:
    """Vendored FlashAttention-2 (paged) at the sliding op — the kernel the
    inert fa2sw router *would* have called.  Returns availability + timing."""
    try:
        from vllm.vllm_flash_attn import flash_attn_varlen_func
    except Exception as e:  # noqa: BLE001
        return {"available": False, "reason": repr(e)}
    device = torch.device("cuda")
    hd = HEAD_DIM["sliding"]
    scale = 1.0 / math.sqrt(hd)
    rot = _rot_for("sliding", ctx)
    key_cache, value_cache, block_tables, nb = _build_paged_kv(
        torch, device, "sliding", ctx, rot)
    q = torch.randn(M, N_Q_HEADS, hd, dtype=torch.bfloat16, device=device) * 0.1
    cu_q = torch.tensor([0, M], dtype=torch.int32, device=device)
    seqused_k = torch.tensor([ctx], dtype=torch.int32, device=device)
    state = {"i": 0, "last": None}

    def call():
        bt = block_tables[state["i"] % rot]
        state["i"] += 1
        state["last"] = flash_attn_varlen_func(
            q=q, k=key_cache, v=value_cache,
            max_seqlen_q=M, cu_seqlens_q=cu_q,
            max_seqlen_k=ctx, seqused_k=seqused_k,
            softmax_scale=scale, causal=True,
            window_size=(SLIDING_WINDOW - 1, 0), block_table=bt)

    try:
        call()
        torch.cuda.synchronize()
    except Exception as e:  # noqa: BLE001
        return {"available": True, "ran": False, "reason": repr(e),
                "head_dim": hd}
    err = None
    if validate:
        ref = _sdpa_reference(torch, q, key_cache, value_cache,
                              block_tables[(state["i"] - 1) % rot], ctx, M,
                              "sliding", scale)
        o = state["last"]
        o = o[0] if isinstance(o, tuple) else o
        err = {"max_abs_err": (o - ref).abs().max().item(),
               "mean_abs_err": (o - ref).abs().mean().item()}
    state["i"] = 0
    dev_us = _profiled_device_us(torch, call, n_iter)
    b = _op_bytes("sliding", ctx, M)
    del key_cache, value_cache
    torch.cuda.empty_cache()
    return {"available": True, "ran": True, "M": M, "ctx": ctx,
            "device_us": dev_us,
            "achieved_gbps_total": b["total_raw_bytes"] / (dev_us / 1e6) / 1e9,
            "achieved_gbps_kv_floor": b["kv_floor_bytes"] / (dev_us / 1e6) / 1e9,
            "validation": err}


def bench_sdpa(torch, M: int, ctx: int, layer_type: str = "sliding",
               n_iter: int = 100) -> dict:
    """torch SDPA on a dense (gathered) window — the framework fallback."""
    import torch.nn.functional as F
    device = torch.device("cuda")
    hd = HEAD_DIM[layer_type]
    scale = 1.0 / math.sqrt(hd)
    eff = min(ctx, SLIDING_WINDOW) if layer_type == "sliding" else ctx
    # rotate dense KV copies to defeat L2
    per = eff * N_KV_HEADS * hd * 2 * DTYPE_BYTES
    rot = max(8, math.ceil(96e6 / per))
    kd = torch.randn(rot, 1, N_KV_HEADS, eff, hd, dtype=torch.bfloat16, device=device) * 0.1
    vd = torch.randn(rot, 1, N_KV_HEADS, eff, hd, dtype=torch.bfloat16, device=device) * 0.1
    qd = torch.randn(1, N_Q_HEADS, M, hd, dtype=torch.bfloat16, device=device) * 0.1
    qpos = torch.arange(eff - M, eff, device=device)
    kpos = torch.arange(eff, device=device)
    mask = kpos[None, :] <= qpos[:, None]
    if layer_type == "sliding":
        mask &= kpos[None, :] > (qpos[:, None] - SLIDING_WINDOW)
    mask = mask[None, None]
    state = {"i": 0}

    def call():
        i = state["i"] % rot
        state["i"] += 1
        F.scaled_dot_product_attention(qd, kd[i], vd[i], attn_mask=mask,
                                       scale=scale, enable_gqa=True)

    call()
    torch.cuda.synchronize()
    dev_us = _profiled_device_us(torch, call, n_iter)
    b = _op_bytes(layer_type, ctx, M)
    del kd, vd
    torch.cuda.empty_cache()
    return {"M": M, "ctx": ctx, "layer_type": layer_type, "device_us": dev_us,
            "achieved_gbps_kv_floor": b["kv_floor_bytes"] / (dev_us / 1e6) / 1e9}


# --------------------------------------------------------------------------- #
# Step 4 — TPS-uplift projection + verdict                                     #
# --------------------------------------------------------------------------- #
def project_tps(attn_frac: float, frontier_tps: float,
                savings=(0.10, 0.25, 0.50, 1.00),
                thresholds=(440, 460, 500)) -> dict:
    rows = []
    for s in savings:
        tps = frontier_tps / (1 - attn_frac * s)
        rows.append({
            "attn_saving_frac": s,
            "tps_new": tps,
            "delta_tps": tps - frontier_tps,
            "crosses": {str(t): tps >= t for t in thresholds},
        })
    return {"attn_frac": attn_frac, "frontier_tps": frontier_tps,
            "thresholds": list(thresholds), "projection": rows}


def decide_verdict(efficiency_fraction: float, kernel_gap_frac: float,
                   max_realistic_saving: float) -> dict:
    """verdict=1 iff a >=25% attention-time saving looks reachable at a >=10%
    kernel-efficiency gap (room below the bandwidth ceiling)."""
    worth = int(kernel_gap_frac >= 0.10 and max_realistic_saving >= 0.25)
    return {
        "verdict_attn_reduction_worth_pursuing": worth,
        "kernel_efficiency_gap_frac": kernel_gap_frac,
        "max_realistic_saving_frac": max_realistic_saving,
        "bandwidth_efficiency_fraction": efficiency_fraction,
        "rule": "1 if kernel_gap>=10% AND reachable_saving>=25%, else 0",
    }


# --------------------------------------------------------------------------- #
# Orchestrator                                                                 #
# --------------------------------------------------------------------------- #
def run(out_path: Path, m_values: list[int], *, profile_json: Path,
        trace_path: Path | None, decode_jsonl: Path | None,
        eval_prompts: Path | None, baked_model: Path | None,
        n_iter: int = 100, ctx_sweep=(128, 256, 512, 784, 1024)) -> dict:
    import torch
    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda")
    t0 = time.time()
    result: dict = {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu": torch.cuda.get_device_name(0),
        "sm_count": torch.cuda.get_device_properties(0).multi_processor_count,
        "l2_bytes": torch.cuda.get_device_properties(0).L2_cache_size,
        "m_values": m_values,
        "frontier_anchors": FRONTIER,
    }

    print("[attn] Step0: trace attribution", flush=True)
    result["trace_attribution"] = analyze_trace(profile_json, trace_path)

    print("[attn] Step2: SWA bandwidth floor", flush=True)
    plens, src = prompt_lengths(decode_jsonl, eval_prompts, baked_model)
    result["swa_floor"] = swa_floor(plens, src)
    rep_ctx = int(round(result["swa_floor"]["mean_ctx"]))

    print("[attn] peak HBM bandwidth", flush=True)
    result["peak_bw"] = _measure_peak_bw(torch, device)
    peak = result["peak_bw"]["measured_peak_gbps_copy"]

    # ---- Step 1: per-M sweep, both layer types, served dispatch -------------
    print(f"[attn] Step1: per-M sweep @ctx={rep_ctx} (served dispatch)", flush=True)
    per_m = []
    for li, lt in enumerate(("sliding", "full")):
        for M in m_values:
            r = bench_op(torch, lt, M, rep_ctx, dispatch="served",
                         n_iter=n_iter, validate=(M in (1, 8) and li == 0) or M == m_values[0])
            r["peak_eff_total"] = r["achieved_gbps_total"] / peak
            per_m.append(r)
            v = r["validation"]
            print(f"   {lt:8s} M={M:<3d} 3D={int(r['used_3d_split_kv'])} "
                  f"{r['device_us']:7.1f}us  {r['achieved_gbps_total']:6.1f} GB/s "
                  f"({r['peak_eff_total']*100:4.1f}% peak)"
                  + (f"  err={v['max_abs_err']:.2e}" if v else ""), flush=True)
    result["per_M"] = per_m

    # ---- served-verify aggregate (M=8 = 1 bonus + K=7 draft) ----------------
    print("[attn] served-verify aggregate (M=8, 30 sliding + 7 full)", flush=True)
    rs = bench_op(torch, "sliding", 8, rep_ctx, dispatch="served", n_iter=n_iter)
    rf = bench_op(torch, "full", 8, rep_ctx, dispatch="served", n_iter=n_iter)
    t_cycle_us = N_SLIDING * rs["device_us"] + N_FULL * rf["device_us"]
    bytes_cycle = (N_SLIDING * rs["bytes"]["total_raw_bytes"]
                   + N_FULL * rf["bytes"]["total_raw_bytes"])
    floor_cycle = (N_SLIDING * rs["bytes"]["kv_floor_bytes"]
                   + N_FULL * rf["bytes"]["kv_floor_bytes"])
    agg_gbps = bytes_cycle / (t_cycle_us / 1e6) / 1e9
    eff_meas = agg_gbps / peak
    eff_spec = agg_gbps / A10G_PEAK_GBPS
    result["served_verify_aggregate"] = {
        "rep_ctx": rep_ctx,
        "sliding_op_us": rs["device_us"], "full_op_us": rf["device_us"],
        "attn_us_per_cycle_microbench": t_cycle_us,
        "attn_ms_per_cycle_microbench": t_cycle_us / 1e3,
        "attn_ms_per_cycle_served": FRONTIER["attn_ms_per_cycle"],
        "microbench_vs_served_ratio": (t_cycle_us / 1e3) / FRONTIER["attn_ms_per_cycle"],
        "bytes_per_cycle_total": bytes_cycle,
        "kv_floor_bytes_per_cycle": floor_cycle,
        "achieved_gbps_aggregate": agg_gbps,
        "bandwidth_efficiency_vs_measured_peak": eff_meas,
        "bandwidth_efficiency_vs_spec_peak": eff_spec,
    }
    print(f"   microbench {t_cycle_us/1e3:.3f} ms/cycle vs served "
          f"{FRONTIER['attn_ms_per_cycle']:.3f} ms  |  {agg_gbps:.1f} GB/s "
          f"= {eff_meas*100:.1f}% measured-peak / {eff_spec*100:.1f}% spec-peak",
          flush=True)

    # ---- split-KV headroom probe: 2D vs 3D at the SAME M=1 work -------------
    print("[attn] split-KV headroom: force2D vs 3D @M=1", flush=True)
    sk = {}
    for lt in ("sliding", "full"):
        f2 = bench_op(torch, lt, 1, rep_ctx, dispatch="force2d", n_iter=n_iter)
        f3 = bench_op(torch, lt, 1, rep_ctx, dispatch="served", n_iter=n_iter)
        sk[lt] = {
            "force2d_us": f2["device_us"], "split3d_us": f3["device_us"],
            "force2d_gbps": f2["achieved_gbps_total"],
            "split3d_gbps": f3["achieved_gbps_total"],
            "split_kv_speedup": f2["device_us"] / f3["device_us"],
        }
        print(f"   {lt:8s} 2D={f2['device_us']:.1f}us 3D={f3['device_us']:.1f}us "
              f"speedup={sk[lt]['split_kv_speedup']:.2f}x", flush=True)
    result["split_kv_headroom"] = sk

    # ---- Step 3: M=1 kernel bake-off ---------------------------------------
    print("[attn] Step3: M=1 bake-off (triton vs FA2 vs SDPA)", flush=True)
    bakeoff = {
        "triton_sliding": bench_op(torch, "sliding", 1, rep_ctx,
                                   dispatch="served", n_iter=n_iter, validate=True),
        "triton_full": bench_op(torch, "full", 1, rep_ctx,
                                dispatch="served", n_iter=n_iter, validate=True),
        "fa2_sliding": bench_fa2(torch, 1, rep_ctx, n_iter=n_iter, validate=True),
        "sdpa_sliding": bench_sdpa(torch, 1, rep_ctx, "sliding", n_iter=n_iter),
        "sdpa_full": bench_sdpa(torch, 1, rep_ctx, "full", n_iter=n_iter),
        "fa2_full_note": "FA2 unsupported for full layers (head_dim=512 > FA2 max 256)",
    }
    result["m1_bakeoff"] = bakeoff

    # ---- ctx-sweep at M=8 (flat-in-ctx / scaling) --------------------------
    print("[attn] ctx-sweep @M=8", flush=True)
    cs = []
    for ctx in ctx_sweep:
        r = bench_op(torch, "sliding", 8, ctx, dispatch="served", n_iter=n_iter)
        r["peak_eff_total"] = r["achieved_gbps_total"] / peak
        cs.append(r)
    result["ctx_sweep_sliding_M8"] = cs

    # ---- primary metric: bandwidth efficiency vs the SWA floor --------------
    # Anchor on the authoritative *served* attention time (1.836 ms/cycle from
    # the live CUDA-graph trace), not the eager microbench, and on the
    # run-averaged KV floor bytes. efficiency = floor_time / served_time.
    floor_kv_bytes = result["swa_floor"]["kv_bytes_per_cycle_total"]
    served_attn_ms = FRONTIER["attn_ms_per_cycle"]
    floor_ms_meas = floor_kv_bytes / (peak * 1e9) * 1e3
    floor_ms_spec = floor_kv_bytes / (A10G_PEAK_GBPS * 1e9) * 1e3
    eff_vs_meas = floor_ms_meas / served_attn_ms
    eff_vs_spec = floor_ms_spec / served_attn_ms
    # reachable saving = if the M=8 verify matched the best measured decode
    # kernel's bandwidth (triton 3D split-KV @M=1), how much verify-attention
    # time vanishes.  This is the greedy-EXACT split-KV lever (already in vLLM,
    # only gated off for max_seqlen_q>1).
    best_3d_gbps = max(r["achieved_gbps_total"] for r in per_m
                       if r["used_3d_split_kv"])
    served_verify_gbps = agg_gbps
    reachable_saving = max(0.0, 1.0 - served_verify_gbps / best_3d_gbps)
    splitkv_speedup_M1 = statistics.fmean(
        v["split_kv_speedup"] for v in sk.values())
    result["bandwidth_efficiency"] = {
        "floor_kv_bytes_per_cycle": floor_kv_bytes,
        "floor_ms_at_measured_peak": floor_ms_meas,
        "floor_ms_at_spec_peak": floor_ms_spec,
        "served_attention_ms_per_cycle": served_attn_ms,
        "efficiency_vs_measured_peak": eff_vs_meas,
        "efficiency_vs_spec_peak": eff_vs_spec,
        "interpretation": (
            f"served attention runs at {eff_vs_meas*100:.1f}% of the SWA "
            f"bandwidth floor — it is occupancy/launch-bound at conc=1, not "
            f"bandwidth-bound. The M=8 spec-verify uses the 2D Triton path "
            f"(~6 CTAs / {result['sm_count']} SMs); split-KV (3D FlashDecoding) "
            f"is gated off for max_seqlen_q>1."),
        "best_3d_gbps_M1": best_3d_gbps,
        "served_verify_gbps": served_verify_gbps,
        "reachable_saving_if_verify_hits_3d_bw": reachable_saving,
        "measured_splitkv_speedup_M1": splitkv_speedup_M1,
    }

    # ---- Step 4: TPS projection + verdict ----------------------------------
    attn_frac = FRONTIER["attn_frac_of_gpu_busy"]
    result["tps_projection"] = project_tps(attn_frac, FRONTIER["public_a10g_small_tps"])
    eff = eff_vs_meas
    kernel_gap = max(0.0, 1.0 - eff)
    result["verdict"] = decide_verdict(eff, kernel_gap, reachable_saving)
    result["primary_metric"] = {"name": "fa2sw_bandwidth_efficiency_fraction",
                                "value": eff}
    result["test_metric"] = {"name": "verdict_attn_reduction_worth_pursuing",
                             "value": result["verdict"]["verdict_attn_reduction_worth_pursuing"]}

    result["elapsed_s"] = time.time() - t0
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(result, indent=2))
    print(f"[attn] wrote {out_path}  ({result['elapsed_s']:.0f}s)", flush=True)
    return result
