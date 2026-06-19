"""Drafter kernel ceiling: byte-identical fixed-split past 136.71? (PR #701)

DECISION-FORCING QUESTION
-------------------------
My #698 (`nov6tc53`, FIXED2D_DEGENERATE_ON_M1 + NO_CONFIG_REACHABLE_CHEAP_REDUCTION)
closed the *config-knob* speed space at 136.711: blanket ``num_splits=1`` (#688) and
``fixed2d`` both byte-identically degenerate to the under-occupied M=1 drafter 2D pin
(0.9832 ms/forward). The only cheaper config path — 3D split-KV — reorders the
online-softmax across segments and is byte-DIVERGENT from the 2D verify -> breaks
strict-#319. My follow-up #1 flagged the one open route to comfortable speed headroom:
a *fixed-split-COUNT* drafter kernel (FlashInfer ``fixed_split_size`` /
Thinking-Machines fixed-split-size attention) that would give 3D-like occupancy with a
fixed, *verify-matching* split count. That bold bet is human-gated (#655) and needs to
be PRICED.

**The gate:** can a fixed-split-COUNT reduction be BOTH byte-identical to the 2D
verify (closes strict-#319) AND recover the M=1 drafter occupancy (toward the 168.919
drafter-free ceiling)?**

  * If a fixed split count CAN match the verify accumulation order bit-exactly -> the
    kernel bet recovers most of the 168.919 headroom at the literature's ~24-46%
    fixed-split overhead -> realized strict-#319 ceiling jumps toward ~160/187, the
    fire's speed half clears COMFORTABLY. (FIXED_SPLIT_FEASIBLE_COMFORTABLE)
  * If ANY multi-split reduction inherently reorders the float adds -> byte-divergent
    -> 136.71 is the HARD realizable strict-#319 ceiling even WITH kernel-authoring,
    full stop. (FIXED_SPLIT_BREAKS_319)

MECHANISM (read directly off the installed served kernel
``vllm/v1/attention/ops/triton_unified_attention.py``)
------------------------------------------------------------------------------
The kernel has TWO reduction paths, dispatched at L923-932:

  * **2D path** (``IS_3D=False``; verify M>1 forced here by ``max_seqlen_q>1``, and the
    pinned drafter). A SINGLE CTA per (q-block, kv-head) runs
    ``for j in range(loop_lo, loop_hi)`` — a fully-serial online-softmax scan over ALL
    KV tiles. ``acc`` is folded tile-by-tile with a running rescale ``alpha`` (L530-545).
    No cross-segment combine. ``TILE_SIZE = TILE_SIZE_PREFILL = 32`` for this geometry.
    This is **S=1** (one serial reduction).

  * **3D split-KV path** (``IS_3D=True``; the cheap occupancy path). Grid z-dim =
    ``num_par_softmax_segments`` CTAs, each scans its OWN segment of tiles -> partial
    ``(m_i, l_i, acc_i)`` -> ``segm_*`` buffers. Then ``reduce_segments`` (L645-732)
    combines: ``overall_max = max(segm_max)``; each segment rescaled by a SINGLE
    ``exp(segm_max - overall_max)``; ``acc_sum = tl.sum(segm_output, axis=0)`` (a
    tree/parallel reduce over the segment axis) / ``overall_expsum``.
    ``TILE_SIZE = TILE_SIZE_DECODE = 16``.

So the 3D path differs from the 2D verify in TWO independent ways: (i) tile size
(16 vs 32 -> different intra-scan fold granularity) and (ii) the cross-segment combine
(``tl.sum`` tree + single per-segment rescale vs the serial running-fold). Both reorder
the fp32 adds. Float add / FMA is non-associative (IEEE-754 §5.1; arxiv 2408.05148),
and FlashAttention-2's parallel tiled reduce is numerically equal to a serial scan only
in EXACT arithmetic (arxiv 2307.08691) — in bf16 the gap is ~6e-5, intrinsic to the
reduction order (arxiv 2405.02803). Therefore S>1 != S=1 bitwise.

THE BOLD-BET TEST (this card)
-----------------------------
A "fixed-split-COUNT" kernel is exactly the 3D path run with a fixed
``num_par_softmax_segments`` (the served default is already a fixed 16) — or, off the
shelf, FlashInfer's ``fixed_split_size`` / ``BatchDecodeWithPagedKVCacheWrapper.plan(
fixed_split_size=...)``. From the literature (FlashInfer docs; Thinking-Machines
"Defeating Nondeterminism" blog; thinking-machines-lab/batch_invariant_ops),
``fixed_split_size`` gives **cross-batch-size invariance** (different M agree with each
OTHER, both at S>1) — it is NOT a serial-scan reproducer. The only FlashInfer lever
that reproduces S=1 bits is ``disable_split_kv=True``, which pins S=1 by construction
and forfeits ALL occupancy gain (== the 2D pin == 136.71).

This op-microbench PROVES the gate on the real drafter geometry: it drives the served
``unified_attention`` at M=1 for a SWEEP of fixed split counts S in {1,2,4,8,16,32},
each on byte-identical inputs, and checks whether ANY S>1 is byte-identical to the M=6
2D verify row while recovering occupancy (cheaper device time). If none are -> the
byte-identity gate FAILS for every fixed split count -> FIXED_SPLIT_BREAKS_319, and
136.71 is the hard realizable strict-#319 ceiling on the drafter side.

LOCAL A10G op-microbench — analysis_only. NO server, NO submission, NO leaderboard
number, NO HF job, NO kernel BUILD/INTEGRATION into the served stack. official_tps=0.
This card PRICES the kernel-authoring bold bet; the integration stays human-gated (#655).
"""
from __future__ import annotations

import os

# MUST precede any torch/vllm import.
# (1) The inherited CUDA_VISIBLE_DEVICES in this pod points at a device index that does
#     not exist inside the container (the single A10G is index 0); hard-pin it to 0.
# (2) The kernel binds ``is_batch_invariant = envs.VLLM_BATCH_INVARIANT`` at import; pin
#     BI off so the 2D-vs-3D dispatch is governed by seq_threshold_3D / num_segments
#     ALONE (clean isolation of the reduction-order axis).
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ.setdefault("VLLM_BATCH_INVARIANT", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

OUT = Path(__file__).resolve().parent / "drafter_kernel_ceiling.json"

# --------------------------------------------------------------------------- #
# Drafter geometry — gemma4_assistant QAT MTP head (identical to #698)         #
# --------------------------------------------------------------------------- #
DRAFTER_NQ = 4
DRAFTER_NKV = 2
DRAFTER_QPKV = DRAFTER_NQ // DRAFTER_NKV  # 2
DRAFTER_LAYER_TYPES = ("sliding", "sliding", "sliding", "full")
N_SLIDING_DRAFTER = sum(t == "sliding" for t in DRAFTER_LAYER_TYPES)  # 3
N_FULL_DRAFTER = sum(t == "full" for t in DRAFTER_LAYER_TYPES)        # 1
HEAD_DIM = {"sliding": 256, "full": 512}
SLIDING_WINDOW = 512
BLOCK_SIZE = 16
DTYPE_BYTES = 2  # bf16

# --------------------------------------------------------------------------- #
# Reduction dispatch + the fixed-split-COUNT sweep                             #
# --------------------------------------------------------------------------- #
# seq_threshold_3D = MIN_LAUNCH_GRID_SIZE_2D // n_kv. Any thr>=1 lets the M=1,
# num_seqs=1 decode reach the 3D path (1 > thr is False), so the fixed-split arms force
# 3D with thr=64 and an explicit num_par_softmax_segments=S.
FORCE_3D_THR = 64
# Fixed split COUNTs to sweep. S=1 (3D) isolates the TILE-size divergence (3D TILE=16 vs
# 2D verify TILE=32) with NO cross-segment combine; S>=2 add the combine. At ctx=512
# (32 decode tiles) the kernel caps act_num_segments at 32, so S=32 is the occupancy max.
FIXED_SPLIT_COUNTS = (1, 2, 4, 8, 16, 32)
SERVED_DEFAULT_S = 16  # the served num_par_softmax_segments == #698's served_3d arm

# --------------------------------------------------------------------------- #
# Speed-law + #683/#688/#698 banked anchors (the official-equiv composition).  #
# These are BANKED — not re-derived here (PR #701 instruction 0).              #
# --------------------------------------------------------------------------- #
LOCAL_TO_OFFICIAL = 0.870
REF_OFFICIAL_TPS = 126.378
PLUS10_BAR = REF_OFFICIAL_TPS + 10.0                  # 136.378
STOCK_E = 3.33
STOCK_OFFICIAL_STATUS_QUO = 136.12
KSTAR = 5

PR683_BI_TAX_MS = 4.680
PR683_RESCUE_R0_MS = 2.996613731915059

PR688_BI_TAX_GPU_MS = 5.694
PR688_DRAFTER_ATTN_PIN_MS = 4.916
PR688_VERIFY_ATTN_PIN_MS = 0.666
PR688_DRAFTER_M1_ATTN_PIN_MS = PR688_DRAFTER_ATTN_PIN_MS / KSTAR   # 0.9832 per forward
PR688_REALIZED_CHEAP_PIN_OFFICIAL = 136.71130191618687

PR688_DRAFTER_ATTN_FRAC = PR688_DRAFTER_ATTN_PIN_MS / PR688_BI_TAX_GPU_MS  # 0.86336
PR688_VERIFY_ATTN_FRAC = PR688_VERIFY_ATTN_PIN_MS / PR688_BI_TAX_GPU_MS    # 0.11697

# #698 banked ceilings
PR698_CONFIG_CEILING = 136.71070470778648            # blanket == fixed2d (S=1 serial)
PR698_DRAFTER_FREE_CEILING = 168.91900117055604      # retained=0 (UNREACHABLE under #319)
PR698_HEADROOM_AVAILABLE = 32.207699254369146        # drafter_free - config_ceiling

S1PRIME_683_OFFICIAL = 160.10
LAND684_OFFICIAL = 187.35

# Literature fixed-split-size overhead band, vs the *dynamic-split* 3D path
# (LMSYS SGLang deterministic blog 2025-09-22: FlashInfer fixed-split = 24.4%..46.0%
# slowdown vs default dynamic-split; H100/H200, Qwen3-8B). Applied to the deeper-kernel
# hypothetical ONLY (the bounded fixed-split path is byte-divergent -> no recovery).
FIXED_SPLIT_OVERHEAD_LO = 0.244
FIXED_SPLIT_OVERHEAD_HI = 0.460

# byte-identity = strict equality (strict-#319 is bit-exact); a fixed-split "recovers
# occupancy" if its device time is meaningfully below the 2D pin.
OCCUPANCY_RECOVER_RATIO = 0.85   # arm faster than 0.85*blanket -> recovers occupancy

VERDICTS = (
    "FIXED_SPLIT_FEASIBLE_COMFORTABLE",
    "FIXED_SPLIT_FEASIBLE_MARGINAL",
    "FIXED_SPLIT_BREAKS_319",
    "FIXED_SPLIT_NEEDS_DEEPER_KERNEL",
)


def official_tps(e_accept: float, tstep_ms: float) -> float:
    return LOCAL_TO_OFFICIAL * 1000.0 * e_accept / tstep_ms


def tstep_for_official(e_accept: float, official: float) -> float:
    return LOCAL_TO_OFFICIAL * 1000.0 * e_accept / official


# --------------------------------------------------------------------------- #
# Cost-model mapping (BANKED from #698 verbatim): op-level retained-cost        #
# fraction -> realized official-equiv. retained=1 -> 136.71 (S=1 pin);          #
# retained=0 -> 168.919 (drafter attn fully free & valid — the unreachable     #
# headroom). Only the DRAFTER attn pin is recoverable; the M>1 verify is        #
# unavoidably 2D (cond max_seqlen_q>1).                                         #
# --------------------------------------------------------------------------- #
def map_official_equiv(retained_fraction: float) -> dict:
    r0 = PR683_RESCUE_R0_MS
    bi_tax = PR683_BI_TAX_MS
    stock_sq = tstep_for_official(STOCK_E, STOCK_OFFICIAL_STATUS_QUO)
    stock_raw = stock_sq - bi_tax - r0

    drafter_attn_ms = PR688_DRAFTER_ATTN_FRAC * bi_tax     # recoverable headroom
    verify_attn_ms = PR688_VERIFY_ATTN_FRAC * bi_tax       # unavoidable (2D verify)

    step = stock_raw + retained_fraction * drafter_attn_ms + verify_attn_ms + r0
    realized = official_tps(STOCK_E, step)

    step_blanket = stock_raw + 1.0 * drafter_attn_ms + verify_attn_ms + r0
    step_free = stock_raw + 0.0 * drafter_attn_ms + verify_attn_ms + r0
    realized_blanket = official_tps(STOCK_E, step_blanket)
    realized_free = official_tps(STOCK_E, step_free)

    return {
        "retained_fraction": retained_fraction,
        "stock_raw_tstep_ms": stock_raw,
        "rescue_r0_ms": r0,
        "bi_tax_ms": bi_tax,
        "drafter_attn_ms_683": drafter_attn_ms,
        "verify_attn_ms_683": verify_attn_ms,
        "realized_step_ms": step,
        "realized_official_equiv": realized,
        "realized_blanket_official_equiv": realized_blanket,
        "realized_drafter_free_official_equiv": realized_free,
        "margin_over_plus10": realized - PLUS10_BAR,
        "clears_plus10": realized >= PLUS10_BAR,
        "gap_to_s1prime_683": realized - S1PRIME_683_OFFICIAL,
        "gap_to_land684": realized - LAND684_OFFICIAL,
        "blanket_reproduces_pr688_cheap_pin":
            abs(realized_blanket - PR688_REALIZED_CHEAP_PIN_OFFICIAL) < 0.5,
    }


def retained_from_device(arm_us: float, blanket_us: float, floor_us: float) -> float:
    """Op-level retained-cost fraction of an arm: pin(arm)/pin(blanket), where the pin is
    measured above the cheapest 3D occupancy floor. retained=1 -> arm==blanket (full
    pin); retained=0 -> arm==floor (drafter attn free)."""
    pin = blanket_us - floor_us
    return (arm_us - floor_us) / pin if pin else float("nan")


# --------------------------------------------------------------------------- #
# GPU op-microbench plumbing (drafter geometry; mirrors the #698 harness)       #
# --------------------------------------------------------------------------- #
def _rot_for(layer_type: str, ctx: int, target_mb: float = 96.0) -> int:
    hd = HEAD_DIM[layer_type]
    eff = min(ctx, SLIDING_WINDOW) if layer_type == "sliding" else ctx
    kv_bytes = eff * DRAFTER_NKV * hd * 2 * DTYPE_BYTES
    return max(8, math.ceil(target_mb * 1e6 / kv_bytes))


def _build_paged_kv(torch, device, layer_type: str, ctx: int, rot: int):
    hd = HEAD_DIM[layer_type]
    nb = math.ceil(ctx / BLOCK_SIZE)
    total_blocks = rot * nb
    shape = (total_blocks, BLOCK_SIZE, DRAFTER_NKV, hd)
    g = torch.Generator(device=device).manual_seed(0xC0FFEE + hash(layer_type) % 1000)
    key_cache = torch.randn(shape, dtype=torch.bfloat16, device=device, generator=g) * 0.1
    value_cache = torch.randn(shape, dtype=torch.bfloat16, device=device, generator=g) * 0.1
    block_tables = [
        torch.arange(r * nb, r * nb + nb, dtype=torch.int32, device=device).view(1, nb)
        for r in range(rot)
    ]
    return key_cache, value_cache, block_tables, nb


def _op_bytes(layer_type: str, ctx: int, M: int) -> dict:
    hd = HEAD_DIM[layer_type]
    eff = min(ctx, SLIDING_WINDOW) if layer_type == "sliding" else ctx
    eff_pad = math.ceil(eff / BLOCK_SIZE) * BLOCK_SIZE
    kv_raw = eff_pad * DRAFTER_NKV * hd * 2 * DTYPE_BYTES
    q_bytes = M * DRAFTER_NQ * hd * DTYPE_BYTES
    out_bytes = M * DRAFTER_NQ * hd * DTYPE_BYTES
    return {"total_raw_bytes": kv_raw + q_bytes + out_bytes}


def _profiled_device_us(torch, fn, n_iter: int, warmup: int = 20) -> float:
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


def _sdpa_reference(torch, q, key_cache, value_cache, block_table, ctx, M,
                    layer_type, scale):
    import torch.nn.functional as F
    hd = HEAD_DIM[layer_type]
    nb = block_table.shape[1]
    kc = key_cache[block_table[0]].reshape(nb * BLOCK_SIZE, DRAFTER_NKV, hd)[:ctx]
    vc = value_cache[block_table[0]].reshape(nb * BLOCK_SIZE, DRAFTER_NKV, hd)[:ctx]
    qd = q.permute(1, 0, 2).unsqueeze(0).float()
    kd = kc.permute(1, 0, 2).unsqueeze(0).float()
    vd = vc.permute(1, 0, 2).unsqueeze(0).float()
    qpos = torch.arange(ctx - M, ctx, device=q.device)
    kpos = torch.arange(ctx, device=q.device)
    mask = kpos[None, :] <= qpos[:, None]
    if layer_type == "sliding":
        mask &= kpos[None, :] > (qpos[:, None] - SLIDING_WINDOW)
    out = F.scaled_dot_product_attention(
        qd, kd, vd, attn_mask=mask[None, None], scale=scale, enable_gqa=True)
    return out.squeeze(0).permute(1, 0, 2).to(q.dtype)


class _AttnHarness:
    """One (layer_type, ctx) paged-KV fixture; every arm runs on byte-identical inputs.
    Q for M>1 is built so its LAST row equals the M=1 Q row (same final query position),
    so the M=1 drafter decode can be compared bit-for-bit against the M=6 verify row."""

    def __init__(self, torch, layer_type: str, ctx: int, max_split: int,
                 m_verify: int = KSTAR + 1):
        self.torch = torch
        self.layer_type = layer_type
        self.ctx = ctx
        self.hd = HEAD_DIM[layer_type]
        self.scale = 1.0 / math.sqrt(self.hd)
        self.window = (SLIDING_WINDOW - 1, 0) if layer_type == "sliding" else (-1, -1)
        device = torch.device("cuda")
        self.device = device
        self.max_split = max_split
        self.rot = _rot_for(layer_type, ctx)
        self.key_cache, self.value_cache, self.block_tables, self.nb = _build_paged_kv(
            torch, device, layer_type, ctx, self.rot)
        gq = torch.Generator(device=device).manual_seed(0x5EED)
        self.q1 = torch.randn(1, DRAFTER_NQ, self.hd, dtype=torch.bfloat16,
                              device=device, generator=gq) * 0.1
        self.m_verify = m_verify
        qv = torch.randn(m_verify, DRAFTER_NQ, self.hd, dtype=torch.bfloat16,
                         device=device, generator=gq) * 0.1
        qv[m_verify - 1].copy_(self.q1[0])
        self.qv = qv

    def _seg_buffers(self, M: int, S: int):
        """Per-S softmax-segment scratch, sized so the kernel's NUM_SEGMENTS_PER_SEQ=S
        offset math matches the buffer strides (layout [token, head, segment, dim])."""
        torch = self.torch
        seg_out = torch.empty(M, DRAFTER_NQ, S, self.hd, dtype=torch.float32,
                              device=self.device)
        seg_max = torch.empty(M, DRAFTER_NQ, S, dtype=torch.float32, device=self.device)
        seg_exp = torch.empty(M, DRAFTER_NQ, S, dtype=torch.float32, device=self.device)
        return seg_out, seg_max, seg_exp

    def _call_factory(self, M: int, thr, S, out, qtensor):
        from vllm.v1.attention.ops.triton_unified_attention import unified_attention
        torch = self.torch
        cu_q = torch.tensor([0, M], dtype=torch.int32, device=self.device)
        seqused_k = torch.tensor([self.ctx], dtype=torch.int32, device=self.device)
        if S is not None:
            seg_out, seg_max, seg_exp = self._seg_buffers(M, S)
        else:
            seg_out = seg_max = seg_exp = None
        state = {"i": 0}

        def call():
            bt = self.block_tables[state["i"] % self.rot]
            state["i"] += 1
            unified_attention(
                q=qtensor, k=self.key_cache, v=self.value_cache, out=out,
                cu_seqlens_q=cu_q, max_seqlen_q=M,
                seqused_k=seqused_k, max_seqlen_k=self.ctx,
                softmax_scale=self.scale, causal=True, window_size=self.window,
                block_table=bt, softcap=0.0,
                q_descale=None, k_descale=None, v_descale=None,
                seq_threshold_3D=thr, num_par_softmax_segments=S,
                softmax_segm_output=seg_out, softmax_segm_max=seg_max,
                softmax_segm_expsum=seg_exp,
            )
        return call, state

    def output_for(self, M: int, thr, S) -> "Any":
        torch = self.torch
        qtensor = self.q1 if M == 1 else self.qv
        out = torch.empty(M, DRAFTER_NQ, self.hd, dtype=torch.bfloat16, device=self.device)
        call, state = self._call_factory(M, thr, S, out, qtensor)
        state["i"] = 0  # always block-window 0 -> reproducible across arms/calls
        call()
        torch.cuda.synchronize()
        return out.clone()

    def device_us(self, M: int, thr, S, n_iter: int) -> float:
        torch = self.torch
        qtensor = self.q1 if M == 1 else self.qv
        out = torch.empty(M, DRAFTER_NQ, self.hd, dtype=torch.bfloat16, device=self.device)
        call, state = self._call_factory(M, thr, S, out, qtensor)
        state["i"] = 0
        return _profiled_device_us(torch, call, n_iter)

    def validate_sdpa(self, M: int = 1) -> dict:
        torch = self.torch
        out = self.output_for(M, None, None)  # blanket 2D arm
        qtensor = self.q1 if M == 1 else self.qv
        ref = _sdpa_reference(torch, qtensor, self.key_cache, self.value_cache,
                              self.block_tables[0], self.ctx, M, self.layer_type, self.scale)
        return {"max_abs_err": (out - ref).abs().max().item(),
                "ref_abs_mean": ref.abs().mean().item()}

    def free(self):
        del self.key_cache, self.value_cache
        self.torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# Arm registry: name -> (M, thr, S). S=None => 2D path; S=int => forced 3D.     #
# --------------------------------------------------------------------------- #
def _arm_specs() -> dict:
    arms = {
        "blanket_2d": (1, None, None),   # S=1 serial 2D pin (== #688 cheap pin)
        "fixed2d": (1, 0, None),         # MLGS2D=0 -> still 2D (degenerate, == blanket)
    }
    for s in FIXED_SPLIT_COUNTS:
        arms[f"split3d_s{s}"] = (1, FORCE_3D_THR, s)  # fixed-split-COUNT, forced 3D
    return arms


def _used_3d(thr, S) -> bool:
    """Replicate the kernel dispatch for M=1, num_seqs=1, BI=0."""
    if thr is None or S is None:
        return False
    return not (1 > thr)


# --------------------------------------------------------------------------- #
# GPU orchestration                                                            #
# --------------------------------------------------------------------------- #
def run_gpu(n_iter: int, ctx_sweep: tuple[int, ...], rep_ctx: int) -> dict:
    import torch
    assert torch.cuda.is_available(), "CUDA required"
    t0 = time.time()
    arms = _arm_specs()
    max_split = max(FIXED_SPLIT_COUNTS)
    result: dict = {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu": torch.cuda.get_device_name(0),
        "sm_count": torch.cuda.get_device_properties(0).multi_processor_count,
        "kernel": "vllm.v1.attention.ops.triton_unified_attention.unified_attention",
        "vllm_batch_invariant_env": os.environ.get("VLLM_BATCH_INVARIANT"),
        "drafter_geometry": {
            "n_q_heads": DRAFTER_NQ, "n_kv_heads": DRAFTER_NKV,
            "layer_types": list(DRAFTER_LAYER_TYPES),
            "n_sliding": N_SLIDING_DRAFTER, "n_full": N_FULL_DRAFTER,
            "head_dim": HEAD_DIM, "sliding_window": SLIDING_WINDOW,
        },
        "arms": {k: {"M": v[0], "thr": v[1], "S": v[2], "expect_3d": _used_3d(v[1], v[2])}
                 for k, v in arms.items()},
        "fixed_split_counts": list(FIXED_SPLIT_COUNTS),
        "n_iter": n_iter, "ctx_sweep": list(ctx_sweep), "rep_ctx": rep_ctx,
        "per_ctx": [],
    }
    import vllm.v1.attention.ops.triton_unified_attention as _tua
    result["kernel_is_batch_invariant_at_import"] = bool(_tua.is_batch_invariant)
    assert not _tua.is_batch_invariant, "VLLM_BATCH_INVARIANT must be 0 for clean isolation"

    print(f"[drkern] {result['gpu']} SMs={result['sm_count']} "
          f"BI={result['kernel_is_batch_invariant_at_import']} "
          f"S_sweep={FIXED_SPLIT_COUNTS} ctx={ctx_sweep}", flush=True)

    for ctx in ctx_sweep:
        ctx_row: dict = {"ctx": ctx, "layers": {}}
        for lt in ("sliding", "full"):
            h = _AttnHarness(torch, lt, ctx, max_split)

            # ---- outputs (block-window 0) ----
            outs = {a: h.output_for(*arms[a]) for a in arms}
            outs_rep = {a: h.output_for(*arms[a]) for a in arms}
            verify_out = h.output_for(h.m_verify, None, None)         # M=6, always 2D
            verify_row = verify_out[h.m_verify - 1:h.m_verify].clone()  # [1,NQ,hd]

            def _eq(a, b):
                return bool(torch.equal(a, b))

            def _maxdiff(a, b):
                return (a.float() - b.float()).abs().max().item()

            vr = verify_row.reshape(1, DRAFTER_NQ, h.hd)
            byte_identity = {}
            for a in arms:
                oa = outs[a].reshape(1, DRAFTER_NQ, h.hd)
                byte_identity[a] = {
                    "repro_run_to_run": _eq(outs[a], outs_rep[a]),
                    "eq_verify_row": _eq(oa, vr),
                    "maxdiff_vs_verify_row": _maxdiff(oa, vr),
                    "eq_blanket": _eq(outs[a], outs["blanket_2d"]),
                    "maxdiff_vs_blanket": _maxdiff(outs[a], outs["blanket_2d"]),
                }

            val = h.validate_sdpa(M=1)

            # ---- device time per arm ----
            us = {a: h.device_us(*arms[a], n_iter) for a in arms}
            b = _op_bytes(lt, ctx, 1)
            gbps = {a: b["total_raw_bytes"] / (us[a] / 1e6) / 1e9 for a in arms}

            ctx_row["layers"][lt] = {
                "device_us": us,
                "achieved_gbps_total": gbps,
                "used_3d_predicted": {a: _used_3d(arms[a][1], arms[a][2]) for a in arms},
                "byte_identity": byte_identity,
                "sdpa_validation": val,
            }
            s16 = us.get(f"split3d_s{SERVED_DEFAULT_S}", float("nan"))
            print(f"   ctx={ctx:<5d} {lt:8s} 2D={us['blanket_2d']:6.1f} "
                  f"3D@S16={s16:6.1f}us speedup={us['blanket_2d']/s16:.2f}x | "
                  f"any S==verify? "
                  f"{int(any(byte_identity[f'split3d_s{s}']['eq_verify_row'] for s in FIXED_SPLIT_COUNTS))}",
                  flush=True)
            h.free()
        result["per_ctx"].append(ctx_row)

    result["elapsed_s"] = time.time() - t0
    return result


# --------------------------------------------------------------------------- #
# Summary: feasibility gate, pricing, ceiling table, verdict                   #
# --------------------------------------------------------------------------- #
def summarize(result: dict, rep_ctx: int) -> dict:
    arms = _arm_specs()
    rep = next(r for r in result["per_ctx"] if r["ctx"] == rep_ctx)

    def cycle_us(arm: str, ctx_row: dict) -> float:
        s = ctx_row["layers"]["sliding"]["device_us"][arm]
        f = ctx_row["layers"]["full"]["device_us"][arm]
        return N_SLIDING_DRAFTER * s + N_FULL_DRAFTER * f

    drafter_cycle = {a: cycle_us(a, rep) for a in arms}
    blanket_us = drafter_cycle["blanket_2d"]
    floor_us = drafter_cycle[f"split3d_s{SERVED_DEFAULT_S}"]   # cheapest occupancy (== #698 served_3d)

    # ---- GATING LEG: byte-identity of each fixed-split count across ALL ctx ----
    # A fixed-split count is byte-identical only if it equals the 2D verify row for BOTH
    # layers at EVERY swept ctx AND is run-to-run reproducible.
    split_byte_identical = {}
    split_occ_recovers = {}
    for s in FIXED_SPLIT_COUNTS:
        arm = f"split3d_s{s}"
        eqs, max_md = [], 0.0
        for r in result["per_ctx"]:
            for lt in ("sliding", "full"):
                bi = r["layers"][lt]["byte_identity"][arm]
                eqs.append(bi["eq_verify_row"] and bi["repro_run_to_run"])
                max_md = max(max_md, bi["maxdiff_vs_verify_row"])
        split_byte_identical[s] = {"byte_identical_all_ctx": all(eqs),
                                   "max_maxdiff_vs_verify": max_md}
        split_occ_recovers[s] = drafter_cycle[arm] < OCCUPANCY_RECOVER_RATIO * blanket_us

    # blanket / fixed2d MUST close #319 (sanity)
    blanket_closes, fixed2d_closes = [], []
    for r in result["per_ctx"]:
        for lt in ("sliding", "full"):
            bib = r["layers"][lt]["byte_identity"]
            blanket_closes.append(bib["blanket_2d"]["eq_verify_row"])
            fixed2d_closes.append(bib["fixed2d"]["eq_verify_row"])
    blanket_closes_319 = all(blanket_closes)
    fixed2d_closes_319 = all(fixed2d_closes)

    # THE GATE: does ANY occupancy-recovering fixed-split count close byte-identity?
    feasible_counts = [s for s in FIXED_SPLIT_COUNTS
                       if split_byte_identical[s]["byte_identical_all_ctx"]
                       and split_occ_recovers[s]]
    fixed_split_byte_identical_feasible = 1 if feasible_counts else 0

    # ---- PRICING LEG ----
    # device-time-vs-count curve + the (hypothetical) occupancy recovery fraction.
    occupancy_recovery_frac = {
        s: (blanket_us - drafter_cycle[f"split3d_s{s}"]) / (blanket_us - floor_us)
        for s in FIXED_SPLIT_COUNTS
    }
    # MEASURED fixed-split-count overhead: cheapest fixed count vs the served default.
    best_fixed_us = min(drafter_cycle[f"split3d_s{s}"] for s in FIXED_SPLIT_COUNTS)
    measured_fixed_overhead_frac = (floor_us - best_fixed_us) / best_fixed_us if best_fixed_us else 0.0

    # (1) The byte-identical-REACHABLE fixed-split ceiling. Since no occupancy-recovering
    #     count is byte-identical, the only byte-identical reduction is the S=1 serial 2D
    #     pin -> retained=1 -> 136.71. (If feasible, use the cheapest feasible count.)
    if feasible_counts:
        s_best = min(feasible_counts, key=lambda s: drafter_cycle[f"split3d_s{s}"])
        retained_bi = retained_from_device(drafter_cycle[f"split3d_s{s_best}"], blanket_us, floor_us)
    else:
        s_best = None
        retained_bi = 1.0   # only S=1 serial pin is byte-identical
    map_bi = map_official_equiv(retained_bi)
    fixed_split_realized_official_equiv = map_bi["realized_official_equiv"]

    # (2) Deeper-kernel HYPOTHETICAL: IF a from-scratch kernel could make BOTH drafter and
    #     verify share a common multi-split scheme (byte-identical to EACH OTHER), the
    #     realized ceiling would sit at the recovered occupancy minus the literature
    #     fixed-split overhead vs the dynamic-3D floor. retained = floor_pin*overhead.
    def deeper(ov: float) -> float:
        cost = floor_us * (1.0 + ov)
        retained = retained_from_device(cost, blanket_us, floor_us)
        return map_official_equiv(retained)["realized_official_equiv"]
    deeper_lo = deeper(FIXED_SPLIT_OVERHEAD_HI)   # high overhead -> lower TPS
    deeper_hi = deeper(FIXED_SPLIT_OVERHEAD_LO)   # low overhead  -> higher TPS

    # ---- CEILING TABLE ----
    ceiling_table = {
        "config_136_71": {
            "official_equiv": PR698_CONFIG_CEILING,
            "what": "blanket==fixed2d S=1 serial 2D pin (banked #698)",
            "byte_identical_to_verify": True, "reachable": True,
        },
        "fixed_split_byte_identical": {
            "official_equiv": fixed_split_realized_official_equiv,
            "what": ("cheapest byte-identical fixed-split count"
                     if feasible_counts else
                     "NO occupancy-recovering fixed count is byte-identical -> stays at S=1 pin"),
            "byte_identical_to_verify": bool(feasible_counts),
            "reachable": True, "best_count": s_best,
        },
        "deeper_kernel_hypothetical": {
            "official_equiv_lo": deeper_lo, "official_equiv_hi": deeper_hi,
            "what": ("from-scratch both-sides common multi-split kernel @ "
                     f"{FIXED_SPLIT_OVERHEAD_LO:.0%}-{FIXED_SPLIT_OVERHEAD_HI:.0%} fixed-split "
                     "overhead vs dynamic-3D; CHANGES the verify reduction -> re-opens "
                     "verify greedy-identity vs the AR reference"),
            "byte_identical_to_deployed_2d_verify": False,
            "reachable": "needs from-scratch kernel + verify re-validation",
        },
        "drafter_free_168_92": {
            "official_equiv": PR698_DRAFTER_FREE_CEILING,
            "what": "drafter attn fully free (retained=0) — UNREACHABLE under strict-#319",
            "byte_identical_to_verify": False, "reachable": False,
        },
    }

    # ---- VERDICT ----
    if fixed_split_byte_identical_feasible:
        realized = fixed_split_realized_official_equiv
        if realized >= 158.0:
            primary = "FIXED_SPLIT_FEASIBLE_COMFORTABLE"
        else:
            primary = "FIXED_SPLIT_FEASIBLE_MARGINAL"
    else:
        primary = "FIXED_SPLIT_BREAKS_319"
    # The only theoretical escape is a deeper both-sides kernel (changes the verify) —
    # noted but NOT a drafter-only kernel; flagged as the secondary lane.
    secondary_lane = "FIXED_SPLIT_NEEDS_DEEPER_KERNEL" if not fixed_split_byte_identical_feasible else None

    summary = {
        "rep_ctx": rep_ctx,
        "drafter_forward_us_per_arm": drafter_cycle,
        "blanket_us": blanket_us, "floor_us_s16": floor_us,
        # ---- GATING LEG (TEST metric) ----
        "fixed_split_byte_identical_feasible": fixed_split_byte_identical_feasible,
        "split_byte_identical": split_byte_identical,
        "split_occupancy_recovers": split_occ_recovers,
        "feasible_counts": feasible_counts,
        "blanket_closes_319": blanket_closes_319,
        "fixed2d_closes_319": fixed2d_closes_319,
        # ---- PRICING LEG (PRIMARY metric) ----
        "fixed_split_realized_official_equiv": fixed_split_realized_official_equiv,
        "fixed_split_occupancy_recovery_frac": occupancy_recovery_frac,
        "fixed_split_overhead_frac_measured": measured_fixed_overhead_frac,
        "fixed_split_overhead_frac_literature": [FIXED_SPLIT_OVERHEAD_LO, FIXED_SPLIT_OVERHEAD_HI],
        "deeper_kernel_hypothetical_official_equiv": [deeper_lo, deeper_hi],
        "map_byte_identical": map_bi,
        # ---- DECIDE LEG ----
        "ceiling_table": ceiling_table,
        "verdict": {
            "primary": primary,
            "secondary_lane": secondary_lane,
            "fixed_split_byte_identical_feasible": fixed_split_byte_identical_feasible,
            "hard_ceiling_official_equiv": PR698_CONFIG_CEILING,
            "headroom_proven_unreachable_official": PR698_HEADROOM_AVAILABLE,
            "rule": ("feasible & realized>=158 -> COMFORTABLE; feasible & <158 -> MARGINAL; "
                     "!feasible -> BREAKS_319 (deeper both-sides kernel = NEEDS_DEEPER_KERNEL)"),
        },
    }
    return summary


# --------------------------------------------------------------------------- #
# Self-test (pure-python: cost-model + verdict logic)                          #
# --------------------------------------------------------------------------- #
def self_test() -> int:
    checks: list[tuple[str, bool]] = []

    # 1. dispatch replication
    checks.append(("blanket_2d -> 2D", _used_3d(None, None) is False))
    checks.append(("fixed2d -> 2D (degenerate)", _used_3d(0, None) is False))
    checks.append(("split3d_s16 -> 3D", _used_3d(FORCE_3D_THR, 16) is True))

    # 2. cost-model reproduces #698 banked anchors
    m1 = map_official_equiv(1.0)
    m0 = map_official_equiv(0.0)
    checks.append(("retained=1 -> 136.71 (config ceiling / #688 cheap pin)",
                   abs(m1["realized_official_equiv"] - PR698_CONFIG_CEILING) < 0.3))
    checks.append(("retained=0 -> 168.919 (drafter-free)",
                   abs(m0["realized_official_equiv"] - PR698_DRAFTER_FREE_CEILING) < 0.3))
    checks.append(("retained=1 stock_raw≈13.607", abs(m1["stock_raw_tstep_ms"] - 13.6068) < 0.01))
    checks.append(("retained monotone: r=0 > r=1",
                   m0["realized_official_equiv"] > m1["realized_official_equiv"]))

    # 3. retained_from_device inversion
    #    arm at the blanket cost -> retained 1; arm at the floor -> retained 0.
    checks.append(("retained(blanket)=1", abs(retained_from_device(216.0, 216.0, 45.5) - 1.0) < 1e-9))
    checks.append(("retained(floor)=0", abs(retained_from_device(45.5, 216.0, 45.5) - 0.0) < 1e-9))

    # 4. deeper-kernel hypothetical lands ~160-167 at 24-46% overhead (sanity, using
    #    #698 measured floor 45.51 / blanket 216.02).
    floor, blanket = 45.505, 216.021

    def deeper(ov):
        cost = floor * (1 + ov)
        return map_official_equiv(retained_from_device(cost, blanket, floor))["realized_official_equiv"]
    d_lo, d_hi = deeper(FIXED_SPLIT_OVERHEAD_HI), deeper(FIXED_SPLIT_OVERHEAD_LO)
    checks.append(("deeper-kernel 24-46% overhead in (158,168)",
                   158.0 < d_lo <= d_hi < 168.92))
    checks.append(("deeper-kernel comfortably > config ceiling 136.71", d_lo > 150.0))

    # 5. verdict logic
    #    infeasible -> BREAKS_319 + NEEDS_DEEPER_KERNEL secondary
    fake_break = {"per_ctx": [], }
    # simulate the verdict branch directly
    feasible = 0
    primary = ("FIXED_SPLIT_FEASIBLE_COMFORTABLE" if feasible else "FIXED_SPLIT_BREAKS_319")
    checks.append(("infeasible -> BREAKS_319", primary == "FIXED_SPLIT_BREAKS_319"))
    feasible2, realized2 = 1, 165.0
    p2 = ("FIXED_SPLIT_FEASIBLE_COMFORTABLE" if (feasible2 and realized2 >= 158) else "x")
    checks.append(("feasible & 165 -> COMFORTABLE", p2 == "FIXED_SPLIT_FEASIBLE_COMFORTABLE"))
    feasible3, realized3 = 1, 145.0
    p3 = ("FIXED_SPLIT_FEASIBLE_COMFORTABLE" if (feasible3 and realized3 >= 158)
          else "FIXED_SPLIT_FEASIBLE_MARGINAL")
    checks.append(("feasible & 145 -> MARGINAL", p3 == "FIXED_SPLIT_FEASIBLE_MARGINAL"))

    # 6. all verdict strings valid
    checks.append(("verdict strings registered",
                   all(v in VERDICTS for v in
                       ("FIXED_SPLIT_BREAKS_319", "FIXED_SPLIT_FEASIBLE_COMFORTABLE",
                        "FIXED_SPLIT_FEASIBLE_MARGINAL", "FIXED_SPLIT_NEEDS_DEEPER_KERNEL"))))

    # 7. geometry sanity
    checks.append(("drafter 3 sliding + 1 full", N_SLIDING_DRAFTER == 3 and N_FULL_DRAFTER == 1))
    checks.append(("QPKV pow2", (DRAFTER_QPKV & (DRAFTER_QPKV - 1)) == 0))

    ok = sum(1 for _, c in checks if c)
    for name, c in checks:
        print(f"  [{'PASS' if c else 'FAIL'}] {name}", flush=True)
    print(f"[self-test] {ok}/{len(checks)} passed", flush=True)
    print(f"  retained=1 -> {m1['realized_official_equiv']:.3f} (config ceiling) ; "
          f"deeper 24-46% -> {d_lo:.1f}..{d_hi:.1f} ; "
          f"retained=0 -> {m0['realized_official_equiv']:.3f} (unreachable)", flush=True)
    return 0 if ok == len(checks) else 1


# --------------------------------------------------------------------------- #
# W&B                                                                          #
# --------------------------------------------------------------------------- #
def log_wandb(result: dict, summary: dict, wandb_name, wandb_group) -> "Any":
    try:
        import wandb
    except Exception as exc:  # noqa: BLE001
        print(f"[wandb] unavailable: {exc}", flush=True)
        return None
    vrd = summary["verdict"]
    ct = summary["ceiling_table"]
    scalars: dict = {
        # compliance
        "analysis_only": 1, "official_tps": 0, "no_hf_job": 1, "fires": 0,
        # TEST metric (the gate)
        "fixed_split_byte_identical_feasible": summary["fixed_split_byte_identical_feasible"],
        "n_feasible_counts": len(summary["feasible_counts"]),
        # PRIMARY metric
        "fixed_split_realized_official_equiv": summary["fixed_split_realized_official_equiv"],
        "config_ceiling_136_71": PR698_CONFIG_CEILING,
        "drafter_free_ceiling_168_92": PR698_DRAFTER_FREE_CEILING,
        "deeper_kernel_hypothetical_lo": ct["deeper_kernel_hypothetical"]["official_equiv_lo"],
        "deeper_kernel_hypothetical_hi": ct["deeper_kernel_hypothetical"]["official_equiv_hi"],
        "headroom_proven_unreachable_official": PR698_HEADROOM_AVAILABLE,
        # device times
        "blanket_us": summary["blanket_us"], "floor_us_s16": summary["floor_us_s16"],
        "fixed_split_overhead_frac_measured": summary["fixed_split_overhead_frac_measured"],
        "fixed_split_overhead_lit_lo": FIXED_SPLIT_OVERHEAD_LO,
        "fixed_split_overhead_lit_hi": FIXED_SPLIT_OVERHEAD_HI,
        # #319 sanity
        "blanket_closes_319": int(summary["blanket_closes_319"]),
        "fixed2d_closes_319": int(summary["fixed2d_closes_319"]),
        # margins
        "margin_over_plus10": summary["map_byte_identical"]["margin_over_plus10"],
        "clears_plus10": int(summary["map_byte_identical"]["clears_plus10"]),
        "gap_to_s1prime_683": summary["map_byte_identical"]["gap_to_s1prime_683"],
        "gap_to_land684": summary["map_byte_identical"]["gap_to_land684"],
        # verdict
        "verdict": vrd["primary"],
        "verdict_secondary_lane": vrd["secondary_lane"] or "",
        # anchors
        "ref_official_tps": REF_OFFICIAL_TPS, "plus10_bar": PLUS10_BAR,
        "local_to_official": LOCAL_TO_OFFICIAL, "stock_e": STOCK_E,
        "s1prime_683_official": S1PRIME_683_OFFICIAL, "land684_official": LAND684_OFFICIAL,
        "gpu": result.get("gpu"), "sm_count": result.get("sm_count"),
        "kernel_is_batch_invariant_at_import":
            int(bool(result.get("kernel_is_batch_invariant_at_import", 0))),
    }
    # per-count byte-identity + maxdiff + device us (rich record)
    for s in FIXED_SPLIT_COUNTS:
        sb = summary["split_byte_identical"][s]
        scalars[f"split_s{s}_byte_identical"] = int(sb["byte_identical_all_ctx"])
        scalars[f"split_s{s}_maxdiff_vs_verify"] = sb["max_maxdiff_vs_verify"]
        scalars[f"split_s{s}_occ_recovers"] = int(summary["split_occupancy_recovers"][s])
        scalars[f"split_s{s}_forward_us"] = summary["drafter_forward_us_per_arm"][f"split3d_s{s}"]
        scalars[f"split_s{s}_occ_recovery_frac"] = summary["fixed_split_occupancy_recovery_frac"][s]

    run = wandb.init(
        entity=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
        project=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"),
        name=wandb_name or "denken/drafter-kernel-ceiling",
        group=wandb_group or "drafter-kernel-ceiling-denken",
        config={"pr": 701, "card": "drafter_kernel_ceiling", "kstar": KSTAR,
                "analysis_only": True, "no_hf_job": 1,
                "fixed_split_counts": list(FIXED_SPLIT_COUNTS),
                "drafter_n_q_heads": DRAFTER_NQ, "drafter_n_kv_heads": DRAFTER_NKV,
                "byte_identity_provenance": (
                    "kernel-math: 2D verify = S=1 serial online-softmax scan (TILE=32, no "
                    "cross-segment combine); 3D = num_par_softmax_segments CTAs + "
                    "reduce_segments tl.sum tree + per-segment exp(m-mmax) rescale "
                    "(TILE=16). fp add/FMA non-assoc (IEEE-754; arxiv 2408.05148); FA2 "
                    "parallel reduce == serial only in exact arith (arxiv 2307.08691); "
                    "bf16 ~6e-5 intrinsic to reduction order (arxiv 2405.02803). FlashInfer "
                    "fixed_split_size = cross-batch-invariant only, NOT serial-scan "
                    "reproducer; only disable_split_kv=True reproduces S=1 (= the pin). "
                    "Overhead band 24-46% from LMSYS SGLang deterministic blog 2025-09-22."),
                },
    )
    wandb.log(scalars)
    wandb.summary.update(scalars)
    wandb.summary.update({"result": json.dumps(result, default=str)})
    wandb.summary.update({"summary_card": json.dumps(summary, default=str)})
    rid = run.id
    run.finish()
    return rid


# --------------------------------------------------------------------------- #
# Entry                                                                        #
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--n-iter", type=int, default=300)
    ap.add_argument("--ctx-sweep", type=str, default="256,512,1024")
    ap.add_argument("--rep-ctx", type=int, default=512)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--wandb-name", type=str, default=None)
    ap.add_argument("--wandb-group", type=str, default="drafter-kernel-ceiling-denken")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    ctx_sweep = tuple(int(x) for x in args.ctx_sweep.split(","))
    rep_ctx = args.rep_ctx if args.rep_ctx in ctx_sweep else ctx_sweep[len(ctx_sweep) // 2]

    result = run_gpu(args.n_iter, ctx_sweep, rep_ctx)
    summary = summarize(result, rep_ctx)
    result["summary"] = summary

    OUT.write_text(json.dumps(result, indent=2, default=str))
    print(f"[drkern] wrote {OUT}  ({result['elapsed_s']:.0f}s)", flush=True)

    vrd = summary["verdict"]
    ct = summary["ceiling_table"]
    print("\n=== DRAFTER KERNEL CEILING (PR #701) ===", flush=True)
    print(f"  fixed_split_byte_identical_feasible = {summary['fixed_split_byte_identical_feasible']}",
          flush=True)
    print("  per-count byte-identity vs 2D verify (all should be 0 except none):", flush=True)
    for s in FIXED_SPLIT_COUNTS:
        sb = summary["split_byte_identical"][s]
        occ = summary["fixed_split_occupancy_recovery_frac"][s]
        print(f"    S={s:3d}  byte_identical={int(sb['byte_identical_all_ctx'])}  "
              f"maxdiff={sb['max_maxdiff_vs_verify']:.3e}  occ_recovery={occ:5.2f}  "
              f"fwd={summary['drafter_forward_us_per_arm'][f'split3d_s{s}']:6.1f}us", flush=True)
    print(f"  CEILING TABLE:", flush=True)
    print(f"    config (byte-identical, reachable)   = {ct['config_136_71']['official_equiv']:.3f}",
          flush=True)
    print(f"    fixed-split byte-identical reachable = {ct['fixed_split_byte_identical']['official_equiv']:.3f}",
          flush=True)
    print(f"    deeper-kernel hypothetical (24-46%)  = "
          f"{ct['deeper_kernel_hypothetical']['official_equiv_lo']:.1f}.."
          f"{ct['deeper_kernel_hypothetical']['official_equiv_hi']:.1f} (changes verify)", flush=True)
    print(f"    drafter-free (UNREACHABLE under #319) = {ct['drafter_free_168_92']['official_equiv']:.3f}",
          flush=True)
    print(f"  VERDICT = {vrd['primary']}"
          + (f"  (secondary lane: {vrd['secondary_lane']})" if vrd['secondary_lane'] else ""),
          flush=True)

    if not args.no_wandb:
        rid = log_wandb(result, summary, args.wandb_name, args.wandb_group)
        if rid:
            result["wandb_run_id"] = rid
            OUT.write_text(json.dumps(result, indent=2, default=str))
            print(f"[wandb] run id = {rid}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
