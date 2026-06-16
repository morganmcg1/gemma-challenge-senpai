"""PR #500 deliverable (1) de-risk: validate the PACKAGED byteexact_splitkv_patch
re-jit mechanism + numerical M-invariance IN-PROCESS, before any full serve run.

Unlike the #496 microbench (which relied on a serve-venv file edit whose kernel
read ``BYTEEXACT_FIXED_TPS`` per call), this drives the *packaged* monkeypatch
(``submissions/fa2sw_strict_byteexact_splitkv399/byteexact_splitkv_patch.py``):
it re-jits the two ``@triton.jit`` kernels on the STOCK wheel via inspect+exec and
raises the backend segment count -- exactly what sitecustomize does when served.

Two modes (run as two processes; the kernel is baked at install, so one T/process):
  --mode fixed : install the packaged patch (T=4, S=64) -> tiles_per_segment baked
                 to 4 -> fixed 64-key chunks at fixed absolute positions. Expect
                 0 flips at straddle positions (byte-exact, M-invariant).
  --mode stock : no patch, deployed nseg=16 adaptive split-KV. Expect >0 flips at
                 the 256-straddle (reproduces the #496 adaptive non-exactness).

Both arms cross the 256 key boundary at straddle base 250 (rows 250..257). Only the
fixed-size scheme keeps every absolute key position in the same segment regardless
of seq_len, so only it is byte-identical between the M=8 verify and the M=1 AR
decode of the same token.

Run (serve venv, GPU 0):
  CUDA_VISIBLE_DEVICES=0 /tmp/senpai-venvs/<hash>/bin/python \
    research/speed/byteexact_attn/verify_packaged_patch.py --mode fixed
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
SUBMISSION_DIR = ROOT / "submissions" / "fa2sw_strict_byteexact_splitkv399"

# Gemma-4-E4B GLOBAL full-attention layer shape (the 7 layers paying the tax).
NUM_Q_HEADS = 8
NUM_KV_HEADS = 2
HEAD_DIM = 512
BLOCK_SIZE = 16
DTYPE = torch.bfloat16
SEQ_THRESHOLD_3D = 64
SCALE = 1.0 / math.sqrt(HEAD_DIM)
WINDOW = (-1, -1)
SOFTCAP = 0.0
DEVICE = "cuda"
NUM_BLOCKS = 256
MAXPOS = NUM_BLOCKS * BLOCK_SIZE  # 4096


def build_static():
    g = torch.Generator(device=DEVICE).manual_seed(1234)
    kcache = torch.randn(NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM,
                         device=DEVICE, dtype=DTYPE, generator=g) * 0.1
    vcache = torch.randn(NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM,
                         device=DEVICE, dtype=DTYPE, generator=g) * 0.1
    qbank = torch.randn(MAXPOS, NUM_Q_HEADS, HEAD_DIM,
                        device=DEVICE, dtype=DTYPE, generator=g) * 0.1
    block_table = torch.arange(NUM_BLOCKS, device=DEVICE, dtype=torch.int32).view(1, NUM_BLOCKS)
    return kcache, vcache, qbank, block_table


def make_segm_buffers(nseg: int):
    segm_out = torch.empty(SEQ_THRESHOLD_3D, NUM_Q_HEADS, nseg, HEAD_DIM, device=DEVICE, dtype=torch.float32)
    segm_max = torch.empty(SEQ_THRESHOLD_3D, NUM_Q_HEADS, nseg, device=DEVICE, dtype=torch.float32)
    segm_exp = torch.empty(SEQ_THRESHOLD_3D, NUM_Q_HEADS, nseg, device=DEVICE, dtype=torch.float32)
    return segm_out, segm_max, segm_exp


def run_attn_3d(ua, static, *, M, base_pos, nseg, seqused_override=None):
    """One 3D split-KV unified_attention call (max_seqlen_q forced to 1, like
    splitkv_verify). M query rows at absolute positions base_pos..base_pos+M-1."""
    kcache, vcache, qbank, block_table = static
    q = qbank[base_pos:base_pos + M].contiguous()
    out = torch.empty(M, NUM_Q_HEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE)
    S = int(base_pos + M) if seqused_override is None else int(seqused_override)
    cu_q = torch.tensor([0, M], device=DEVICE, dtype=torch.int32)
    seqused = torch.tensor([S], device=DEVICE, dtype=torch.int32)
    segm_out, segm_max, segm_exp = make_segm_buffers(nseg)
    ua.unified_attention(
        q=q, k=kcache, v=vcache, out=out,
        cu_seqlens_q=cu_q, max_seqlen_q=1, seqused_k=seqused, max_seqlen_k=S,
        softmax_scale=SCALE, causal=True, window_size=WINDOW,
        block_table=block_table, softcap=SOFTCAP,
        q_descale=None, k_descale=None, v_descale=None,
        seq_threshold_3D=SEQ_THRESHOLD_3D, num_par_softmax_segments=nseg,
        softmax_segm_output=segm_out, softmax_segm_max=segm_max, softmax_segm_expsum=segm_exp,
    )
    return out


def m_invariance(ua, static, *, base_pos, M, nseg):
    out_verify = run_attn_3d(ua, static, M=M, base_pos=base_pos, nseg=nseg)
    rows = []
    for i in range(M):
        pos = base_pos + i
        out_ar = run_attn_3d(ua, static, M=1, base_pos=pos, nseg=nseg, seqused_override=pos + 1)
        eq_bytes = torch.equal(out_verify[i].view(torch.int16), out_ar[0].view(torch.int16))
        max_abs = float((out_verify[i].float() - out_ar[0].float()).abs().max().item())
        rows.append({"row": i, "abs_pos": pos, "byte_equal": bool(eq_bytes), "max_abs_err": max_abs})
    flips = sum(0 if r["byte_equal"] else 1 for r in rows)
    return {"flips": flips, "n_rows": M, "rows": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fixed", "adaptive"], required=True)
    ap.add_argument("--nseg", type=int, default=64,
                    help="num_par_softmax_segments (fixed-mode candidate uses 64)")
    ap.add_argument("--fixed-tps", type=int, default=4)
    ap.add_argument("--bases", default="250:straddle256,506:straddle512,2042:straddle2048,100:control",
                    help="comma list of base:tag (M=8 verify rows base..base+7)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "need GPU (CUDA_VISIBLE_DEVICES=0)"
    print(f"device: {torch.cuda.get_device_name(0)} mode={args.mode} nseg={args.nseg}", flush=True)

    rejit = {"installed": False, "fixed_tps_marker": None, "kernel_has_literal": None,
             "kernel_has_adaptive": None, "reduce_has_literal": None,
             "backend_num_segments": None}

    if args.mode == "fixed":
        os.environ["BYTEEXACT_FIXED_TPS"] = str(args.fixed_tps)
        os.environ["BYTEEXACT_NUM_SEGMENTS"] = str(args.nseg)
        sys.path.insert(0, str(SUBMISSION_DIR))
        import byteexact_splitkv_patch as bx
        rejit["installed"] = bool(bx.install())

    import vllm.v1.attention.ops.triton_unified_attention as ua
    import vllm.v1.attention.backends.triton_attn as ta

    lit = f"tiles_per_segment = {args.fixed_tps}"
    ksrc = inspect.getsource(getattr(ua.kernel_unified_attention, "fn", ua.kernel_unified_attention))
    rsrc = inspect.getsource(getattr(ua.reduce_segments, "fn", ua.reduce_segments))
    rejit["fixed_tps_marker"] = getattr(ua, "_byteexact_fixed_tps", None)
    rejit["kernel_has_literal"] = (lit in ksrc)
    rejit["kernel_has_adaptive"] = ("tiles_per_segment = cdiv_fn(seq_len, NUM_SEGMENTS_PER_SEQ * TILE_SIZE)" in ksrc)
    rejit["reduce_has_literal"] = (lit in rsrc)
    rejit["backend_num_segments"] = getattr(ta, "NUM_PAR_SOFTMAX_SEGMENTS", None)
    print(f"[verify] re-jit introspection: {json.dumps(rejit)}", flush=True)

    bases = []
    for tok in args.bases.split(","):
        b, _, tag = tok.partition(":")
        bases.append((tag or f"base{b}", int(b)))

    static = build_static()
    _ = run_attn_3d(ua, static, M=1, base_pos=100, nseg=args.nseg)  # warm/compile
    torch.cuda.synchronize()

    proof, straddle_flips, control_flips = {}, 0, 0
    for tag, base in bases:
        res = m_invariance(ua, static, base_pos=base, M=8, nseg=args.nseg)
        proof[tag] = res
        if "control" in tag:
            control_flips += res["flips"]
        else:
            straddle_flips += res["flips"]
        print(f"[verify] {args.mode:8s} {tag:12s} base={base:4d} nseg={args.nseg:2d} "
              f"flips={res['flips']}/{res['n_rows']} "
              f"max_abs_err={max(r['max_abs_err'] for r in res['rows']):.2e}", flush=True)

    result = {"mode": args.mode, "nseg": args.nseg, "fixed_tps": args.fixed_tps,
              "rejit": rejit, "proof": proof,
              "straddle_flips_total": straddle_flips, "control_flips": control_flips}

    if args.mode == "fixed":
        ok = (rejit["installed"] and rejit["kernel_has_literal"] and not rejit["kernel_has_adaptive"]
              and rejit["reduce_has_literal"] and rejit["backend_num_segments"] == args.nseg
              and straddle_flips == 0 and control_flips == 0)
        result["pass"] = bool(ok)
        print(f"[verify] FIXED verdict: re-jit installed={rejit['installed']} "
              f"literal_baked={rejit['kernel_has_literal']}/{rejit['reduce_has_literal']} "
              f"adaptive_gone={not rejit['kernel_has_adaptive']} backend_seg={rejit['backend_num_segments']} "
              f"straddle_flips={straddle_flips} control_flips={control_flips} (want 0/0) -> PASS={ok}", flush=True)
    else:
        result["pass"] = None
        print(f"[verify] ADAPTIVE contrast (nseg={args.nseg}): straddle_flips={straddle_flips} "
              f"(>0 demonstrates the adaptive split-KV is NOT byte-exact -> the fixed lever matters)", flush=True)

    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"[verify] wrote {args.out}", flush=True)
    return 0 if (args.mode == "adaptive" or result.get("pass")) else 1


if __name__ == "__main__":
    sys.exit(main())
