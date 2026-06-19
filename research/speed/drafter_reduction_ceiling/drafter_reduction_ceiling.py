"""Drafter fixed2d vs blanket pin: is 136.71 the strict-#319 spec ceiling? (PR #698)

DECISION-FORCING QUESTION
-------------------------
Land #684/#691 closed the strict-#319 greedy break by forcing the drafter's M=1
decode attention onto the deterministic 2D (single-serial-softmax) reduction. #688
measured that blanket ``num_splits=1`` pin at ``drafter_m1_attn_pin = 0.9832 ms``
per M=1 forward (4.916 ms summed over the K=5 drafter forwards), giving a realized
strict-#319 spec ceiling of ``realized_cheap_pin_official_equiv = 136.711`` — which
FALLS SHORT of #683's S1'=160 lane and land #684's 187.35 ceiling.

Land #684/#691 also shipped a ``fixed2d`` knob (``MIN_LAUNCH_GRID_SIZE_2D=0``). The
PR asks: does ``fixed2d`` cost LESS than blanket ``num_splits=1`` on the M=1 drafter
decode — recovering headroom toward 160/187.35 — or does it DEGENERATE to
``num_splits=1`` on M=1 (so 136.71 is the config-reachable strict-#319 ceiling and
beating it needs a custom deterministic-multi-split drafter kernel)?

MECHANISM (read directly off the installed served kernel — isolation forbids
reading land #697's branch, so the knob is derived from the kernel, not the serve
config; see PR comment)
------------------------------------------------------------------------------
``vllm/v1/attention/ops/triton_unified_attention.py`` line 923-932 launches 2D iff::

    use_3d = not ( seq_threshold_3D is None
                   or num_par_softmax_segments is None or <segm buffers None>
                   or max_seqlen_q > 1
                   or num_seqs > seq_threshold_3D
                   or is_batch_invariant )

The served backend sets ``seq_threshold_3D = MIN_LAUNCH_GRID_SIZE_2D // num_kv_heads``.
At the M=1 drafter decode (``max_seqlen_q=1``, ``num_seqs=1``), with
``VLLM_BATCH_INVARIANT=0`` (we set it before import so ``is_batch_invariant`` is
False and dispatch is governed by ``seq_threshold_3D`` ALONE):

  (a) blanket num_splits=1  : ``thr=None``  -> use_3d=False (cond 1)            2D
  (b) fixed2d (MLGS2D=0)     : ``thr=0``     -> ``num_seqs(1) > 0`` True (cond 3) 2D  <- DEGENERATE
  (c) un-pinned autotuned    : ``thr=64``    -> ``num_seqs(1) > 64`` False        3D  <- break floor

So fixed2d (thr=0) forces the IDENTICAL 2D path as the blanket on the single-seq
M=1 drafter decode: ``1 > 0`` is always True. It recovers no occupancy. The cheap
3D split-KV path (c) exists but its KV reduction is byte-DIVERGENT from the 2D
verify reduction (M=6 verify is always 2D, cond ``max_seqlen_q>1``), so it BREAKS
strict-#319. There is no config-reachable deterministic-multi-split: a fixed
split-COUNT reduction that matches the 2D verify bytes would need a custom kernel
(FlashInfer ``fixed_split_size`` / Thinking-Machines fixed-split-size, 24-46%
overhead) — NOT a vLLM Triton config knob.

This op-microbench PROVES the mechanism on drafter geometry: it drives the real
served ``unified_attention`` kernel at M=1 for all three arms on byte-identical
inputs, measures device time per arm, checks (i) each arm is run-to-run
reproducible, (ii) fixed2d == blanket byte-for-byte (closes #319), (iii) served_3d
DIVERGES from blanket (the break), (iv) the M=1 2D output reproduces the M=6 verify
2D output for the matching query row (direct #319 closure), then maps the op-level
retained-cost fraction onto #688's banked server pin and #683's bi_tax basis to get
``realized_fixed2d_official_equiv``.

LOCAL op-microbench — no server, no submission, no leaderboard number, no HF job.
"""
from __future__ import annotations

import os

# MUST precede any vllm import: the kernel binds is_batch_invariant=envs.
# VLLM_BATCH_INVARIANT at import. We isolate the seq_threshold_3D dispatch axis by
# pinning BI off, so 2D-vs-3D is governed by the MIN_LAUNCH_GRID_SIZE_2D knob alone.
os.environ.setdefault("VLLM_BATCH_INVARIANT", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

OUT = Path(__file__).resolve().parent / "drafter_reduction_ceiling.json"

# --------------------------------------------------------------------------- #
# Drafter geometry — gemma4_assistant QAT MTP head                            #
# (config.json: num_hidden_layers=4, layer_types=[sliding,sliding,sliding,full],#
#  head_dim=256 / global_head_dim=512, num_attention_heads=4,                  #
#  num_key_value_heads=2, sliding_window=512)                                  #
# --------------------------------------------------------------------------- #
DRAFTER_NQ = 4
DRAFTER_NKV = 2
DRAFTER_QPKV = DRAFTER_NQ // DRAFTER_NKV  # 2 (pow2 -> USE_TD_QO path OK)
DRAFTER_LAYER_TYPES = ("sliding", "sliding", "sliding", "full")
N_SLIDING_DRAFTER = sum(t == "sliding" for t in DRAFTER_LAYER_TYPES)  # 3
N_FULL_DRAFTER = sum(t == "full" for t in DRAFTER_LAYER_TYPES)        # 1
HEAD_DIM = {"sliding": 256, "full": 512}
SLIDING_WINDOW = 512
BLOCK_SIZE = 16
DTYPE_BYTES = 2  # bf16

# --------------------------------------------------------------------------- #
# Reduction dispatch knob: seq_threshold_3D = MIN_LAUNCH_GRID_SIZE_2D // n_kv  #
# --------------------------------------------------------------------------- #
MIN_LAUNCH_GRID_SIZE_2D_DEFAULT = 128
SEQ_THRESHOLD_3D_DEFAULT = MIN_LAUNCH_GRID_SIZE_2D_DEFAULT // DRAFTER_NKV  # 64
N_SEG = 16  # num_par_softmax_segments (served default)

# arm -> seq_threshold_3D passed to unified_attention at M=1, num_seqs=1
ARMS: dict[str, int | None] = {
    "blanket_2d": None,                      # (a) num_splits=1 (= #688 BI=1 pin)
    "fixed2d": 0,                            # (b) MIN_LAUNCH_GRID_SIZE_2D=0
    "served_3d": SEQ_THRESHOLD_3D_DEFAULT,   # (c) un-pinned autotuned break floor
}
# expected 2D/3D per arm at M=1, num_seqs=1 (from the kernel dispatch above)
ARM_EXPECT_3D = {"blanket_2d": False, "fixed2d": False, "served_3d": True}

# --------------------------------------------------------------------------- #
# Speed-law + #683/#688 banked anchors (the official-equiv composition)        #
# --------------------------------------------------------------------------- #
LOCAL_TO_OFFICIAL = 0.870
REF_OFFICIAL_TPS = 126.378
PLUS10_BAR = REF_OFFICIAL_TPS + 10.0                  # 136.378
STOCK_E = 3.33
STOCK_OFFICIAL_STATUS_QUO = 136.12                    # #677 stock spec official
KSTAR = 5

PR683_BI_TAX_MS = 4.680
PR683_RESCUE_R0_MS = 2.996613731915059

# #688 measured per-component pins (run xkuylfj1, int4_mtp local gpu-busy basis):
PR688_BI_TAX_GPU_MS = 5.694
PR688_DRAFTER_ATTN_PIN_MS = 4.916          # drafter_gpu(attn_only) - drafter_gpu(bi_off)
PR688_VERIFY_ATTN_PIN_MS = 0.666           # verify_gpu(attn_only)  - verify_gpu(bi_off)
PR688_DRAFTER_M1_ATTN_PIN_MS = PR688_DRAFTER_ATTN_PIN_MS / KSTAR   # 0.9832 per forward
PR688_ATTN_PIN_REMOVABLE_FRAC = 0.9803301721109944
PR688_REALIZED_CHEAP_PIN_OFFICIAL = 136.71130191618687
PR688_REALIZED_DET_FREE_OFFICIAL = 164.47352274159078

# basis-portable fractions of the blanket bi_tax (drafter / verify attention split)
PR688_DRAFTER_ATTN_FRAC = PR688_DRAFTER_ATTN_PIN_MS / PR688_BI_TAX_GPU_MS  # 0.86336
PR688_VERIFY_ATTN_FRAC = PR688_VERIFY_ATTN_PIN_MS / PR688_BI_TAX_GPU_MS    # 0.11697

S1PRIME_683_OFFICIAL = 160.10
LAND684_OFFICIAL = 187.35

# verdict thresholds on the op-level retained-cost fraction
RETAINED_DEGEN_HI = 0.85   # >= 0.85 of the blanket pin retained -> degenerate (same 2D)
RETAINED_CHEAP_LO = 0.50   # <= 0.50 retained -> fixed2d genuinely cheaper on the drafter

VERDICTS = (
    "FIXED2D_CHEAPER_ON_DRAFTER",
    "FIXED2D_DEGENERATE_ON_M1",
    "NO_CONFIG_REACHABLE_CHEAP_REDUCTION",
    "FIXED2D_BREAKS_319_ON_DRAFTER",
)


def official_tps(e_accept: float, tstep_ms: float) -> float:
    return LOCAL_TO_OFFICIAL * 1000.0 * e_accept / tstep_ms


def tstep_for_official(e_accept: float, official: float) -> float:
    return LOCAL_TO_OFFICIAL * 1000.0 * e_accept / official


# --------------------------------------------------------------------------- #
# GPU op-microbench plumbing (drafter geometry; mirrors the validated          #
# scripts/local_validation/profile_attention.bench_op buffer/profiler pattern) #
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
    kv_floor = eff * DRAFTER_NKV * hd * 2 * DTYPE_BYTES
    kv_raw = eff_pad * DRAFTER_NKV * hd * 2 * DTYPE_BYTES
    q_bytes = M * DRAFTER_NQ * hd * DTYPE_BYTES
    out_bytes = M * DRAFTER_NQ * hd * DTYPE_BYTES
    return {"kv_floor_bytes": kv_floor, "kv_raw_bytes": kv_raw,
            "total_raw_bytes": kv_raw + q_bytes + out_bytes}


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
    """Holds a single (layer_type, ctx) paged-KV bench fixture so all three arms
    run on byte-identical inputs (KV, Q, block tables). Q for M>1 is built so its
    LAST row equals the M=1 Q row (same final query position) -> the 2D path's
    M-flatness lets us prove the drafter M=1 decode reproduces the verify row."""

    def __init__(self, torch, layer_type: str, ctx: int, m_verify: int = KSTAR + 1):
        self.torch = torch
        self.layer_type = layer_type
        self.ctx = ctx
        self.hd = HEAD_DIM[layer_type]
        self.scale = 1.0 / math.sqrt(self.hd)
        self.window = (SLIDING_WINDOW - 1, 0) if layer_type == "sliding" else (-1, -1)
        device = torch.device("cuda")
        self.device = device
        self.rot = _rot_for(layer_type, ctx)
        self.key_cache, self.value_cache, self.block_tables, self.nb = _build_paged_kv(
            torch, device, layer_type, ctx, self.rot)
        gq = torch.Generator(device=device).manual_seed(0x5EED)
        self.q1 = torch.randn(1, DRAFTER_NQ, self.hd, dtype=torch.bfloat16,
                              device=device, generator=gq) * 0.1
        # M=verify query whose LAST row == the M=1 query row (same final position)
        self.m_verify = m_verify
        qv = torch.randn(m_verify, DRAFTER_NQ, self.hd, dtype=torch.bfloat16,
                         device=device, generator=gq) * 0.1
        qv[m_verify - 1].copy_(self.q1[0])
        self.qv = qv
        # 3D split-KV softmax scratch (sized like the served backend)
        seg_rows = max(SEQ_THRESHOLD_3D_DEFAULT, 1)
        self.segm_out = torch.empty(seg_rows, DRAFTER_NQ, N_SEG, self.hd,
                                    dtype=torch.float32, device=device)
        self.segm_max = torch.empty(seg_rows, DRAFTER_NQ, N_SEG,
                                    dtype=torch.float32, device=device)
        self.segm_exp = torch.empty(seg_rows, DRAFTER_NQ, N_SEG,
                                    dtype=torch.float32, device=device)

    def _call_factory(self, M: int, thr: int | None, out, qtensor):
        from vllm.v1.attention.ops.triton_unified_attention import unified_attention
        torch = self.torch
        cu_q = torch.tensor([0, M], dtype=torch.int32, device=self.device)
        seqused_k = torch.tensor([self.ctx], dtype=torch.int32, device=self.device)
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
                seq_threshold_3D=thr, num_par_softmax_segments=N_SEG,
                softmax_segm_output=self.segm_out, softmax_segm_max=self.segm_max,
                softmax_segm_expsum=self.segm_exp,
            )
        return call, state

    def output_for_arm(self, arm: str, M: int = 1, fixed_block: bool = True):
        """One deterministic forward on block-table window 0; returns cloned out."""
        torch = self.torch
        thr = ARMS[arm]
        qtensor = self.q1 if M == 1 else self.qv
        out = torch.empty(M, DRAFTER_NQ, self.hd, dtype=torch.bfloat16, device=self.device)
        call, state = self._call_factory(M, thr, out, qtensor)
        if fixed_block:
            state["i"] = 0  # always window 0 -> reproducible across arms/calls
        call()
        torch.cuda.synchronize()
        return out.clone()

    def device_us(self, arm: str, n_iter: int, M: int = 1) -> float:
        torch = self.torch
        thr = ARMS[arm]
        qtensor = self.q1 if M == 1 else self.qv
        out = torch.empty(M, DRAFTER_NQ, self.hd, dtype=torch.bfloat16, device=self.device)
        call, state = self._call_factory(M, thr, out, qtensor)
        state["i"] = 0
        return _profiled_device_us(torch, call, n_iter)

    def validate_sdpa(self, arm: str, M: int = 1) -> dict:
        torch = self.torch
        out = self.output_for_arm(arm, M=M)
        qtensor = self.q1 if M == 1 else self.qv
        ref = _sdpa_reference(torch, qtensor, self.key_cache, self.value_cache,
                              self.block_tables[0], self.ctx, M, self.layer_type, self.scale)
        return {"max_abs_err": (out - ref).abs().max().item(),
                "ref_abs_mean": ref.abs().mean().item()}

    def free(self):
        del self.key_cache, self.value_cache
        self.torch.cuda.empty_cache()


def _used_3d(arm: str) -> bool:
    """Replicate the kernel dispatch for M=1, num_seqs=1, BI=0."""
    thr = ARMS[arm]
    if thr is None:
        return False
    return not (1 > thr)  # num_seqs(1) > thr ?  -> 2D if True


# --------------------------------------------------------------------------- #
# Mapping: op-level retained-cost fraction -> realized_fixed2d_official_equiv  #
# --------------------------------------------------------------------------- #
def map_official_equiv(retained_fraction: float) -> dict:
    """Transport the op-microbench retained-cost fraction onto #688's banked
    server pin (#683 bi_tax basis), reproducing realized_cheap_pin at retained=1.

    Only the DRAFTER attention pin is recoverable by a cheaper M=1 reduction; the
    M=6 verify is unavoidably 2D (cond max_seqlen_q>1), so its pin stays. The GEMM
    override is dropped (Marlin is byte-identical across M, land #680 -> the
    realized cheap-pin deployment is attn-2D-only, no aten-GEMM override)."""
    r0 = PR683_RESCUE_R0_MS
    bi_tax = PR683_BI_TAX_MS
    stock_sq = tstep_for_official(STOCK_E, STOCK_OFFICIAL_STATUS_QUO)
    stock_raw = stock_sq - bi_tax - r0

    drafter_attn_ms = PR688_DRAFTER_ATTN_FRAC * bi_tax     # recoverable headroom
    verify_attn_ms = PR688_VERIFY_ATTN_FRAC * bi_tax       # unavoidable (2D verify)

    step_fixed2d = stock_raw + retained_fraction * drafter_attn_ms + verify_attn_ms + r0
    realized_fixed2d = official_tps(STOCK_E, step_fixed2d)

    # retained=1 -> blanket ceiling (== #688 realized_cheap_pin); retained=0 ->
    # drafter reduction fully free & valid (the headroom fixed2d fails to recover).
    step_blanket = stock_raw + 1.0 * drafter_attn_ms + verify_attn_ms + r0
    step_drafter_free = stock_raw + 0.0 * drafter_attn_ms + verify_attn_ms + r0
    realized_blanket = official_tps(STOCK_E, step_blanket)
    realized_drafter_free = official_tps(STOCK_E, step_drafter_free)

    return {
        "retained_fraction": retained_fraction,
        "stock_raw_tstep_ms": stock_raw,
        "rescue_r0_ms": r0,
        "bi_tax_ms": bi_tax,
        "drafter_attn_ms_683": drafter_attn_ms,
        "verify_attn_ms_683": verify_attn_ms,
        "realized_fixed2d_step_ms": step_fixed2d,
        "realized_fixed2d_official_equiv": realized_fixed2d,
        "realized_blanket_official_equiv": realized_blanket,
        "realized_drafter_free_official_equiv": realized_drafter_free,
        "headroom_recovered_official": realized_fixed2d - realized_blanket,
        "headroom_available_official": realized_drafter_free - realized_blanket,
        "margin_over_plus10": realized_fixed2d - PLUS10_BAR,
        "clears_plus10": realized_fixed2d >= PLUS10_BAR,
        "gap_to_s1prime_683": realized_fixed2d - S1PRIME_683_OFFICIAL,
        "gap_to_land684": realized_fixed2d - LAND684_OFFICIAL,
        # cross-check that the mapping reproduces the #688 banked ceiling at r=1
        "blanket_reproduces_pr688_cheap_pin": abs(realized_blanket - PR688_REALIZED_CHEAP_PIN_OFFICIAL) < 0.5,
    }


def decide_verdict(retained_fraction: float, fixed2d_closes_319: bool,
                   served_3d_breaks_319: bool) -> dict:
    if not fixed2d_closes_319:
        primary = "FIXED2D_BREAKS_319_ON_DRAFTER"
    elif retained_fraction <= RETAINED_CHEAP_LO:
        primary = "FIXED2D_CHEAPER_ON_DRAFTER"
    elif retained_fraction >= RETAINED_DEGEN_HI:
        primary = "FIXED2D_DEGENERATE_ON_M1"
    else:
        primary = "FIXED2D_DEGENERATE_ON_M1"  # middling still = no usable headroom
    # corollary: degenerate + the only cheaper config (3D) breaks #319
    no_config_cheap = primary == "FIXED2D_DEGENERATE_ON_M1" and served_3d_breaks_319
    return {
        "primary": primary,
        "no_config_reachable_cheap_reduction": no_config_cheap,
        "corollary": ("NO_CONFIG_REACHABLE_CHEAP_REDUCTION" if no_config_cheap else None),
        "retained_fraction": retained_fraction,
        "fixed2d_closes_319": fixed2d_closes_319,
        "served_3d_breaks_319": served_3d_breaks_319,
        "rule": (f"closes_319 & retained>={RETAINED_DEGEN_HI} -> DEGENERATE; "
                 f"closes_319 & retained<={RETAINED_CHEAP_LO} -> CHEAPER; "
                 f"!closes_319 -> BREAKS_319"),
    }


# --------------------------------------------------------------------------- #
# GPU orchestration                                                            #
# --------------------------------------------------------------------------- #
def run_gpu(n_iter: int, ctx_sweep: tuple[int, ...], rep_ctx: int) -> dict:
    import torch
    assert torch.cuda.is_available(), "CUDA required"
    t0 = time.time()
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
        "arms": {k: v for k, v in ARMS.items()},
        "arm_expect_3d": ARM_EXPECT_3D,
        "n_iter": n_iter, "ctx_sweep": list(ctx_sweep), "rep_ctx": rep_ctx,
        "per_ctx": [],
    }
    # import-time BI binding proof
    import vllm.v1.attention.ops.triton_unified_attention as _tua
    result["kernel_is_batch_invariant_at_import"] = bool(_tua.is_batch_invariant)
    assert not _tua.is_batch_invariant, "VLLM_BATCH_INVARIANT must be 0 for clean isolation"

    print(f"[drceil] {result['gpu']} SMs={result['sm_count']} "
          f"BI={result['kernel_is_batch_invariant_at_import']} ctx={ctx_sweep}", flush=True)

    for ctx in ctx_sweep:
        ctx_row: dict = {"ctx": ctx, "layers": {}}
        for lt in ("sliding", "full"):
            h = _AttnHarness(torch, lt, ctx)
            # ---- byte-identity / reproducibility (all on block-window 0) ----
            outs = {arm: h.output_for_arm(arm, M=1) for arm in ARMS}
            outs_rep = {arm: h.output_for_arm(arm, M=1) for arm in ARMS}
            verify_out = h.output_for_arm("blanket_2d", M=h.m_verify)  # M=6, always 2D
            verify_row = verify_out[h.m_verify - 1:h.m_verify].clone()  # last row [1,NQ,hd]

            def _eq(a, b):
                return bool(torch.equal(a, b))

            def _maxdiff(a, b):
                return (a.float() - b.float()).abs().max().item()

            repro = {arm: _eq(outs[arm], outs_rep[arm]) for arm in ARMS}
            # fixed2d == blanket (both 2D -> closes #319 identically)
            fixed2d_eq_blanket = _eq(outs["fixed2d"], outs["blanket_2d"])
            # served_3d vs blanket (3D vs 2D -> the strict-#319 break)
            served3d_eq_blanket = _eq(outs["served_3d"], outs["blanket_2d"])
            # direct #319 closure: M=1 2D decode reproduces the M=6 verify row
            fixed2d_eq_verify_row = _eq(outs["fixed2d"].reshape(1, DRAFTER_NQ, h.hd),
                                        verify_row.reshape(1, DRAFTER_NQ, h.hd))
            served3d_eq_verify_row = _eq(outs["served_3d"].reshape(1, DRAFTER_NQ, h.hd),
                                         verify_row.reshape(1, DRAFTER_NQ, h.hd))

            # ---- numerical sanity vs SDPA (blanket 2D arm) ----
            val = h.validate_sdpa("blanket_2d", M=1)

            # ---- device time per arm ----
            us = {arm: h.device_us(arm, n_iter, M=1) for arm in ARMS}
            b = _op_bytes(lt, ctx, 1)
            gbps = {arm: b["total_raw_bytes"] / (us[arm] / 1e6) / 1e9 for arm in ARMS}

            ctx_row["layers"][lt] = {
                "device_us": us,
                "achieved_gbps_total": gbps,
                "used_3d_predicted": {a: _used_3d(a) for a in ARMS},
                "split_kv_speedup_blanket_over_3d": us["blanket_2d"] / us["served_3d"],
                "fixed2d_vs_blanket_us_ratio": us["fixed2d"] / us["blanket_2d"],
                "byte_identity": {
                    "repro_run_to_run": repro,
                    "fixed2d_eq_blanket": fixed2d_eq_blanket,
                    "served3d_eq_blanket": served3d_eq_blanket,
                    "served3d_vs_blanket_maxdiff": _maxdiff(outs["served_3d"], outs["blanket_2d"]),
                    "fixed2d_eq_verify_row": fixed2d_eq_verify_row,
                    "served3d_eq_verify_row": served3d_eq_verify_row,
                    "fixed2d_vs_verify_row_maxdiff": _maxdiff(
                        outs["fixed2d"].reshape(1, DRAFTER_NQ, h.hd),
                        verify_row.reshape(1, DRAFTER_NQ, h.hd)),
                },
                "sdpa_validation": val,
            }
            print(f"   ctx={ctx:<5d} {lt:8s} "
                  f"2D={us['blanket_2d']:6.1f} fixed2d={us['fixed2d']:6.1f} "
                  f"3D={us['served_3d']:6.1f}us  speedup={us['blanket_2d']/us['served_3d']:.2f}x "
                  f"| f2d==blanket={int(fixed2d_eq_blanket)} 3d==blanket={int(served3d_eq_blanket)} "
                  f"f2d==verifyrow={int(fixed2d_eq_verify_row)}", flush=True)
            h.free()
        result["per_ctx"].append(ctx_row)

    result["elapsed_s"] = time.time() - t0
    return result


def summarize(result: dict, rep_ctx: int) -> dict:
    """Build the per-reduction drafter M=1 cost table, retained fraction, the
    official-equiv mapping, the #319-closure determination and the verdict."""
    rep = next(r for r in result["per_ctx"] if r["ctx"] == rep_ctx)

    def cycle_us(arm: str, ctx_row: dict) -> float:
        s = ctx_row["layers"]["sliding"]["device_us"][arm]
        f = ctx_row["layers"]["full"]["device_us"][arm]
        return N_SLIDING_DRAFTER * s + N_FULL_DRAFTER * f

    # one M=1 drafter forward through all 4 attention layers, per arm
    drafter_cycle = {arm: cycle_us(arm, rep) for arm in ARMS}
    blanket_pin_us = drafter_cycle["blanket_2d"] - drafter_cycle["served_3d"]
    fixed2d_pin_us = drafter_cycle["fixed2d"] - drafter_cycle["served_3d"]
    retained_fraction = (fixed2d_pin_us / blanket_pin_us) if blanket_pin_us else float("nan")

    # ---- #319 closure across all swept ctx (must hold everywhere) ----
    closes = []
    breaks = []
    for r in result["per_ctx"]:
        for lt in ("sliding", "full"):
            bi = r["layers"][lt]["byte_identity"]
            closes.append(bi["fixed2d_eq_blanket"] and bi["fixed2d_eq_verify_row"]
                          and bi["repro_run_to_run"]["fixed2d"])
            breaks.append(not bi["served3d_eq_blanket"])
    fixed2d_closes_319 = all(closes)
    served_3d_breaks_319 = all(breaks)

    mapping = map_official_equiv(retained_fraction)
    verdict = decide_verdict(retained_fraction, fixed2d_closes_319, served_3d_breaks_319)

    # primary metric on #688's server basis: per-forward drafter M=1 attention pin
    fixed2d_drafter_m1_attn_pin_ms = retained_fraction * PR688_DRAFTER_M1_ATTN_PIN_MS
    fixed2d_drafter_attn_pin_ms = retained_fraction * PR688_DRAFTER_ATTN_PIN_MS  # K-summed

    table = {
        "reduction_option": {
            "blanket_num_splits_1": {
                "thr": ARMS["blanket_2d"], "path": "2D",
                "drafter_forward_us": drafter_cycle["blanket_2d"],
                "drafter_m1_attn_pin_ms_server": PR688_DRAFTER_M1_ATTN_PIN_MS,
                "closes_319": True,
            },
            "fixed2d_MLGS2D_0": {
                "thr": ARMS["fixed2d"], "path": "2D (degenerate at num_seqs=1)",
                "drafter_forward_us": drafter_cycle["fixed2d"],
                "drafter_m1_attn_pin_ms_server": fixed2d_drafter_m1_attn_pin_ms,
                "closes_319": fixed2d_closes_319,
            },
            "unpinned_autotuned_3d": {
                "thr": ARMS["served_3d"], "path": "3D split-KV (break floor)",
                "drafter_forward_us": drafter_cycle["served_3d"],
                "drafter_m1_attn_pin_ms_server": 0.0,
                "closes_319": not served_3d_breaks_319,  # False
            },
        },
        "blanket_pin_us": blanket_pin_us,
        "fixed2d_pin_us": fixed2d_pin_us,
        "retained_fraction": retained_fraction,
    }

    summary = {
        "rep_ctx": rep_ctx,
        "drafter_forward_us_per_arm": drafter_cycle,
        "cost_table": table,
        # ---- PRIMARY metric ----
        "fixed2d_drafter_m1_attn_pin_ms": fixed2d_drafter_m1_attn_pin_ms,
        "blanket_drafter_m1_attn_pin_ms": PR688_DRAFTER_M1_ATTN_PIN_MS,
        "fixed2d_drafter_attn_pin_ms_ksummed": fixed2d_drafter_attn_pin_ms,
        "blanket_drafter_attn_pin_ms_ksummed": PR688_DRAFTER_ATTN_PIN_MS,
        "retained_fraction": retained_fraction,
        # ---- #319 closure ----
        "fixed2d_closes_319": fixed2d_closes_319,
        "served_3d_breaks_319": served_3d_breaks_319,
        # ---- TEST metric: realized official-equiv ----
        "mapping": mapping,
        "realized_fixed2d_official_equiv": mapping["realized_fixed2d_official_equiv"],
        # ---- verdict ----
        "verdict": verdict,
    }
    return summary


# --------------------------------------------------------------------------- #
# Self-test (pure-python: mapping reproduces #688; verdict logic)              #
# --------------------------------------------------------------------------- #
def self_test() -> int:
    checks: list[tuple[str, bool]] = []

    # 1. kernel dispatch replication for the three arms at M=1, num_seqs=1
    checks.append(("blanket_2d -> 2D", _used_3d("blanket_2d") is False))
    checks.append(("fixed2d -> 2D (degenerate)", _used_3d("fixed2d") is False))
    checks.append(("served_3d -> 3D", _used_3d("served_3d") is True))

    # 2. mapping reproduces #688 banked ceiling at retained=1.0
    m1 = map_official_equiv(1.0)
    checks.append(("retained=1 blanket==#688 cheap_pin (136.711)",
                   abs(m1["realized_blanket_official_equiv"] - PR688_REALIZED_CHEAP_PIN_OFFICIAL) < 0.3))
    checks.append(("retained=1 fixed2d==blanket",
                   abs(m1["realized_fixed2d_official_equiv"] - m1["realized_blanket_official_equiv"]) < 1e-6))
    checks.append(("retained=1 stock_raw≈13.607",
                   abs(m1["stock_raw_tstep_ms"] - 13.6068) < 0.01))

    # 3. retained=0 (hypothetical valid 3D drafter) recovers real headroom
    m0 = map_official_equiv(0.0)
    checks.append(("retained=0 > retained=1 (headroom exists if valid)",
                   m0["realized_fixed2d_official_equiv"] > m1["realized_fixed2d_official_equiv"]))
    checks.append(("retained=0 drafter_free>160",
                   m0["realized_fixed2d_official_equiv"] > 160.0))

    # 4. monotonicity: more retained cost -> lower official equiv
    seq = [map_official_equiv(r)["realized_fixed2d_official_equiv"]
           for r in (0.0, 0.25, 0.5, 0.75, 1.0)]
    checks.append(("official_equiv monotone decreasing in retained",
                   all(seq[i] > seq[i + 1] for i in range(len(seq) - 1))))

    # 5. verdict logic
    v_degen = decide_verdict(1.0, True, True)
    checks.append(("retained=1,closes,breaks -> DEGENERATE",
                   v_degen["primary"] == "FIXED2D_DEGENERATE_ON_M1"))
    checks.append(("DEGENERATE+3D-breaks -> NO_CONFIG_REACHABLE corollary",
                   v_degen["no_config_reachable_cheap_reduction"] is True))
    v_cheap = decide_verdict(0.2, True, True)
    checks.append(("retained=0.2,closes -> CHEAPER",
                   v_cheap["primary"] == "FIXED2D_CHEAPER_ON_DRAFTER"))
    v_break = decide_verdict(1.0, False, True)
    checks.append(("!closes_319 -> BREAKS_319",
                   v_break["primary"] == "FIXED2D_BREAKS_319_ON_DRAFTER"))

    # 6. expected ceiling: degenerate fixed2d lands at 136.71 (< plus10 136.378? no, >)
    checks.append(("degenerate realized==136.71 < S1'=160",
                   m1["realized_fixed2d_official_equiv"] < S1PRIME_683_OFFICIAL))
    checks.append(("degenerate realized clears +10 bar (136.378)",
                   m1["realized_fixed2d_official_equiv"] >= PLUS10_BAR))

    # 7. geometry sanity
    checks.append(("drafter 3 sliding + 1 full", N_SLIDING_DRAFTER == 3 and N_FULL_DRAFTER == 1))
    checks.append(("QPKV pow2", (DRAFTER_QPKV & (DRAFTER_QPKV - 1)) == 0))

    ok = sum(1 for _, c in checks if c)
    for name, c in checks:
        print(f"  [{'PASS' if c else 'FAIL'}] {name}", flush=True)
    print(f"[self-test] {ok}/{len(checks)} passed", flush=True)
    print(f"  retained=1.0 -> realized_fixed2d={m1['realized_fixed2d_official_equiv']:.3f} "
          f"(ceiling); retained=0.0 -> drafter_free={m0['realized_fixed2d_official_equiv']:.3f}",
          flush=True)
    return 0 if ok == len(checks) else 1


# --------------------------------------------------------------------------- #
# W&B                                                                          #
# --------------------------------------------------------------------------- #
def log_wandb(result: dict, summary: dict, wandb_name: str | None,
              wandb_group: str | None) -> str | None:
    try:
        import wandb
    except Exception as exc:  # noqa: BLE001
        print(f"[wandb] unavailable: {exc}", flush=True)
        return None
    vrd = summary["verdict"]
    mp = summary["mapping"]
    scalars: dict = {
        # compliance (analysis-only, no leaderboard number, no HF job)
        "analysis_only": 1, "official_tps": 0, "no_hf_job": 1, "fires": 0,
        # PRIMARY metric
        "fixed2d_drafter_m1_attn_pin_ms": summary["fixed2d_drafter_m1_attn_pin_ms"],
        "blanket_drafter_m1_attn_pin_ms": summary["blanket_drafter_m1_attn_pin_ms"],
        "fixed2d_drafter_attn_pin_ms_ksummed": summary["fixed2d_drafter_attn_pin_ms_ksummed"],
        "retained_fraction": summary["retained_fraction"],
        "blanket_pin_us": summary["cost_table"]["blanket_pin_us"],
        "fixed2d_pin_us": summary["cost_table"]["fixed2d_pin_us"],
        # per-reduction drafter M=1 device time (op-microbench)
        "drafter_forward_us_blanket_2d": summary["drafter_forward_us_per_arm"]["blanket_2d"],
        "drafter_forward_us_fixed2d": summary["drafter_forward_us_per_arm"]["fixed2d"],
        "drafter_forward_us_served_3d": summary["drafter_forward_us_per_arm"]["served_3d"],
        # #319 closure
        "fixed2d_closes_319": int(summary["fixed2d_closes_319"]),
        "served_3d_breaks_319": int(summary["served_3d_breaks_319"]),
        # TEST metric + headroom decomposition
        "realized_fixed2d_official_equiv": mp["realized_fixed2d_official_equiv"],
        "realized_blanket_official_equiv": mp["realized_blanket_official_equiv"],
        "realized_drafter_free_official_equiv": mp["realized_drafter_free_official_equiv"],
        "headroom_recovered_official": mp["headroom_recovered_official"],
        "headroom_available_official": mp["headroom_available_official"],
        "realized_fixed2d_margin_over_plus10": mp["margin_over_plus10"],
        "realized_fixed2d_clears_plus10": int(mp["clears_plus10"]),
        "gap_to_s1prime_683": mp["gap_to_s1prime_683"],
        "gap_to_land684": mp["gap_to_land684"],
        "blanket_reproduces_pr688_cheap_pin": int(mp["blanket_reproduces_pr688_cheap_pin"]),
        # verdict
        "verdict": vrd["primary"],
        "no_config_reachable_cheap_reduction": int(vrd["no_config_reachable_cheap_reduction"]),
        "verdict_corollary": vrd["corollary"] or "",
        # anchors / provenance
        "ref_official_tps": REF_OFFICIAL_TPS, "plus10_bar": PLUS10_BAR,
        "local_to_official": LOCAL_TO_OFFICIAL, "stock_e": STOCK_E,
        "pr688_realized_cheap_pin": PR688_REALIZED_CHEAP_PIN_OFFICIAL,
        "pr688_realized_det_free": PR688_REALIZED_DET_FREE_OFFICIAL,
        "pr683_bi_tax_ms": PR683_BI_TAX_MS,
        "s1prime_683_official": S1PRIME_683_OFFICIAL,
        "land684_official": LAND684_OFFICIAL,
        "gpu": result.get("gpu"), "sm_count": result.get("sm_count"),
        "kernel_is_batch_invariant_at_import": int(bool(result.get("kernel_is_batch_invariant_at_import", 0))),
    }
    run = wandb.init(
        entity=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
        project=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"),
        name=wandb_name or "denken/drafter-reduction-ceiling",
        group=wandb_group or "drafter-reduction-ceiling-denken",
        config={"pr": 698, "card": "drafter_reduction_ceiling", "kstar": KSTAR,
                "analysis_only": True, "no_hf_job": 1,
                "drafter_n_q_heads": DRAFTER_NQ, "drafter_n_kv_heads": DRAFTER_NKV},
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
    ap.add_argument("--wandb-group", type=str, default="drafter-reduction-ceiling-denken")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    ctx_sweep = tuple(int(x) for x in args.ctx_sweep.split(","))
    rep_ctx = args.rep_ctx if args.rep_ctx in ctx_sweep else ctx_sweep[len(ctx_sweep) // 2]

    result = run_gpu(args.n_iter, ctx_sweep, rep_ctx)
    summary = summarize(result, rep_ctx)
    result["summary"] = summary

    OUT.write_text(json.dumps(result, indent=2, default=str))
    print(f"[drceil] wrote {OUT}  ({result['elapsed_s']:.0f}s)", flush=True)

    vrd = summary["verdict"]
    mp = summary["mapping"]
    print("\n=== DRAFTER REDUCTION CEILING (PR #698) ===", flush=True)
    print(f"  retained_fraction         = {summary['retained_fraction']:.4f}", flush=True)
    print(f"  fixed2d_drafter_m1_pin_ms = {summary['fixed2d_drafter_m1_attn_pin_ms']:.4f} "
          f"(blanket {PR688_DRAFTER_M1_ATTN_PIN_MS:.4f})", flush=True)
    print(f"  fixed2d_closes_319        = {summary['fixed2d_closes_319']}", flush=True)
    print(f"  served_3d_breaks_319      = {summary['served_3d_breaks_319']}", flush=True)
    print(f"  realized_fixed2d_official = {mp['realized_fixed2d_official_equiv']:.3f} "
          f"(blanket {mp['realized_blanket_official_equiv']:.3f}, "
          f"drafter-free {mp['realized_drafter_free_official_equiv']:.3f})", flush=True)
    print(f"  gap_to_s1prime_683        = {mp['gap_to_s1prime_683']:+.3f}", flush=True)
    print(f"  VERDICT                   = {vrd['primary']}"
          + (f" + {vrd['corollary']}" if vrd['corollary'] else ""), flush=True)

    if not args.no_wandb:
        rid = log_wandb(result, summary, args.wandb_name, args.wandb_group)
        if rid:
            result["wandb_run_id"] = rid
            OUT.write_text(json.dumps(result, indent=2, default=str))
            print(f"[wandb] run id = {rid}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
