#!/usr/bin/env python3
"""PR #759: turn two Kineto device traces (BI=0, BI=1) into a per-kernel-FAMILY
batch-invariance tax ledger.

We sum GPU device time (cat in {kernel, gpu_memcpy, gpu_memset}) by kernel name,
classify each name into one of the advisor's families, normalize per completion
token (acceptance is ~equal across arms, #750: 3.331 vs 3.321), then attribute the
BI-induced slowdown: added_ms[family] = bi1_ms[family] - bi0_ms[family]; the
family share = added_ms[family] / sum(added_ms over families with added>0).

Why a family rollup (not a name diff): BI=1 *swaps* kernels (cuBLAS GEMM ->
Triton matmul_kernel_persistent; 3D split-KV attn + reduce_segments -> 2D serial
attn; fused rms_norm -> Triton _rms_norm_kernel). The replacement has a different
name, so only the per-family SUM nets the swap into a single added-ms number.

Headline = relative shares (robust to profiler overhead). Absolute ms/token is
secondary (profiler inflates it vs the official-anchored 1/tps).
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import defaultdict
from pathlib import Path

# Ordered (first match wins). Case-insensitive search on the device kernel name.
#
# Ordering matters because (a) BI=1 *swaps* kernels and (b) inductor fuses ops
# into names like ``triton_red_fused_add_marlin_gemm_mul_rms_norm`` -- that token
# contains "marlin_gemm" but the kernel is the RMSNorm+residual+epilogue GLUE, not
# the heavy matmul (the matmul is the separate ``marlin::Marlin`` kernel). So the
# inductor-fused-glue rule must beat the gemm rule, while the two genuine BI Triton
# matmuls (matmul_kernel_persistent / bmm_kernel, which have no "fused" token) must
# beat the fused-glue rule. attention also beats gemm so the 3D reduce_segments /
# reshape_and_cache aren't stolen by the reduction/copy rules.
FAMILY_RULES = [
    # 1. attention (Triton unified attn, 3D split-KV seg reduction, KV write, rope)
    ("attention", re.compile(
        r"unified_attention|reduce_segments|paged.?attn|flash.?attn|fmha|"
        r"_attn|attn_|attention|reshape_and_cache|concat_and_cache|kv_cache|"
        r"rotary|rope", re.I)),
    # 2. genuine BI Triton matmul (deterministic persistent GEMM / bmm) -- must
    #    precede the inductor-fused rule (these are real matmuls, not glue).
    ("matmul_gemm", re.compile(r"matmul_kernel_persistent|bmm_kernel", re.I)),
    # 3. inductor-fused pointwise/reduction glue -> norm/elementwise (RMSNorm,
    #    residual add, activation, dequant epilogue). Beats the gemm rule so the
    #    "marlin_gemm" token in fused names doesn't steal them into GEMM.
    ("norm_elementwise", re.compile(r"triton_(poi|red|per|tem|for|mm|ext)\w*fused"
                                    r"|triton_\w*fused", re.I)),
    # 4. genuine matmul/GEMM kernels (int4 Marlin body, cuBLAS/cutlass bf16, gemv)
    ("matmul_gemm", re.compile(
        r"marlin|gptq|awq|machete|cutlass|ampere|sm80|sm86|sm90|wgmma|wmma|"
        r"cublas|gemv|s16816|h16816|i16832|splitkreduce|xmma|implicit_gemm|"
        r"\bgemm\b|tile_scheduler", re.I)),
    # 5. sampling / lm_head-side (top-k, softmax, argmax, logits)
    ("sampling_lmhead", re.compile(
        r"gathertopk|sbtopk|top_?k|top_?p|sampl|_log_softmax_kernel|log_softmax|"
        r"softmax|argmax|multinomial|renorm|logits|lm_head|vocab", re.I)),
    # 6. norm / elementwise (non-fused): rms_norm, activations, scale/quant
    ("norm_elementwise", re.compile(
        r"_rms_norm_kernel|rms_?norm|layer_?norm|fused_add_rms|silu|gelu|swiglu|"
        r"geglu|activation|act_and_mul|elementwise|vectorized_elementwise|"
        r"add_kernel|mul_kernel|div_kernel|sub_kernel|exp_kernel|clamp", re.I)),
    # 7. reduction / all-reduce (TP=1 -> ~none; BI deterministic mean)
    ("allreduce_reduction", re.compile(
        r"all_?reduce|nccl|allgather|reduce_scatter|device_reduce|cub::device|"
        r"reduce_kernel|reducekernel|mean_kernel", re.I)),
    # 8. data movement (copy/gather/index/embedding/cast/memcpy) -> other_movement
    ("other_movement", re.compile(
        r"vectorized_gather|index_|index_select|embedding|scatter|\bgather\b|"
        r"cast|convert|_to_copy|copy_kernel|\bcopy\b|memcpy|memset|fill_|arange|"
        r"\bwhere\b|cat_|concat|pad_|narrow|slice", re.I)),
]
DEVICE_CATS = {"kernel", "gpu_memcpy", "gpu_memset", "memcpy", "memset"}


def classify(name: str) -> str:
    for fam, rx in FAMILY_RULES:
        if rx.search(name):
            return fam
    return "other"


def load_events(path: str):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        data = json.load(f)
    return data.get("traceEvents", data if isinstance(data, list) else [])


def arm_kernel_ms(trace_files: list[str]) -> tuple[dict, dict]:
    """Return (per_kernel_ms, per_kernel_count) summed over all device events."""
    per_kernel = defaultdict(float)
    per_count = defaultdict(int)
    for tf in trace_files:
        for ev in load_events(tf):
            if ev.get("ph") != "X":
                continue
            cat = str(ev.get("cat", "")).lower()
            if cat not in DEVICE_CATS:
                continue
            dur = ev.get("dur")
            if not isinstance(dur, (int, float)):
                continue
            name = ev.get("name", "?")
            per_kernel[name] += dur / 1000.0  # us -> ms
            per_count[name] += 1
    return dict(per_kernel), dict(per_count)


def roll_families(per_kernel: dict) -> dict:
    fam = defaultdict(float)
    for name, ms in per_kernel.items():
        fam[classify(name)] += ms
    return dict(fam)


# Within matmul_gemm, split the int4 Marlin body GEMM (a custom CUDA op vLLM's BI
# mode does NOT override) from the bf16 GEMMs (lm_head + drafter) that BI swaps
# cuBLAS->Triton-persistent. This is the load-bearing control: if the GEMM-family
# tax lives entirely in bf16 and ~0 in int4, the BI tax is not the quantization.
_INT4_RX = re.compile(r"marlin|gptq|awq|machete", re.I)
_BF16GEMM_RX = re.compile(
    r"matmul_kernel_persistent|bmm_kernel|ampere|cutlass|wmma|cublas|gemv|"
    r"s16816|h16816|i16832|splitkreduce|xmma|\bgemm\b", re.I)


def matmul_subsplit(per_kernel: dict) -> dict:
    int4 = bf16 = 0.0
    for name, ms in per_kernel.items():
        if classify(name) != "matmul_gemm":
            continue
        if _INT4_RX.search(name):
            int4 += ms
        else:
            bf16 += ms
    return {"gemm_int4_body_ms": round(int4, 4), "gemm_bf16_ms": round(bf16, 4)}


def top_kernels_by_family(per_kernel: dict, n=6) -> dict:
    by_fam = defaultdict(list)
    for name, ms in per_kernel.items():
        by_fam[classify(name)].append((name, ms))
    out = {}
    for f, lst in by_fam.items():
        out[f] = sorted(lst, key=lambda x: -x[1])[:n]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bi0-summary", required=True)
    ap.add_argument("--bi1-summary", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    s0 = json.loads(Path(args.bi0_summary).read_text())
    s1 = json.loads(Path(args.bi1_summary).read_text())
    tok0 = s0.get("completion_tokens") or 1
    tok1 = s1.get("completion_tokens") or 1

    pk0, pc0 = arm_kernel_ms(s0["trace_files"])
    pk1, pc1 = arm_kernel_ms(s1["trace_files"])
    fam0 = roll_families(pk0)
    fam1 = roll_families(pk1)

    families = sorted(set(fam0) | set(fam1))
    # per-output-token ms (acceptance ~equal -> comparable)
    fam0_pt = {f: fam0.get(f, 0.0) / tok0 for f in families}
    fam1_pt = {f: fam1.get(f, 0.0) / tok1 for f in families}
    added_pt = {f: fam1_pt[f] - fam0_pt[f] for f in families}

    total_added_pos = sum(v for v in added_pt.values() if v > 0)
    total_added_net = sum(added_pt.values())

    ledger = []
    for f in families:
        share = (added_pt[f] / total_added_pos) if total_added_pos > 0 else 0.0
        ledger.append({
            "family": f,
            "bi0_ms_per_tok": round(fam0_pt[f], 5),
            "bi1_ms_per_tok": round(fam1_pt[f], 5),
            "added_ms_per_tok": round(added_pt[f], 5),
            "share_of_total_added": round(share, 4),
        })
    ledger.sort(key=lambda r: -r["added_ms_per_tok"])

    tot0 = sum(fam0_pt.values())
    tot1 = sum(fam1_pt.values())
    top = ledger[0] if ledger else {"family": None, "share_of_total_added": 0.0}

    # int4-body vs bf16 GEMM control (per output token). The int4 Marlin body is a
    # custom CUDA op BI does NOT override -> it should ~cancel in the diff; the BI
    # GEMM tax should live almost entirely in the bf16 swap (lm_head + drafter).
    sub0 = matmul_subsplit(pk0)
    sub1 = matmul_subsplit(pk1)
    gemm_int4_added_pt = sub1["gemm_int4_body_ms"] / tok1 - sub0["gemm_int4_body_ms"] / tok0
    gemm_bf16_added_pt = sub1["gemm_bf16_ms"] / tok1 - sub0["gemm_bf16_ms"] / tok0

    out = {
        "completion_tokens_bi0": tok0,
        "completion_tokens_bi1": tok1,
        "total_device_ms_per_tok_bi0": round(tot0, 5),
        "total_device_ms_per_tok_bi1": round(tot1, 5),
        "total_device_ms_per_tok_added": round(tot1 - tot0, 5),
        "profiled_device_bi_tax_pct": round((tot1 - tot0) / tot1, 4) if tot1 else None,
        "total_added_ms_per_tok_positive": round(total_added_pos, 5),
        "total_added_ms_per_tok_net": round(total_added_net, 5),
        "bi_tax_top_op_family": top["family"],
        "bi_tax_top_op_share": top["share_of_total_added"],
        "matmul_subsplit_bi0": sub0,
        "matmul_subsplit_bi1": sub1,
        "gemm_int4_body_added_ms_per_tok": round(gemm_int4_added_pt, 5),
        "gemm_bf16_added_ms_per_tok": round(gemm_bf16_added_pt, 5),
        "ledger": ledger,
        "top_kernels_bi0": {f: [[n, round(m, 4)] for n, m in v]
                            for f, v in top_kernels_by_family(pk0).items()},
        "top_kernels_bi1": {f: [[n, round(m, 4)] for n, m in v]
                            for f, v in top_kernels_by_family(pk1).items()},
        "decode_tps_proxy_bi0": s0.get("decode_tps_proxy"),
        "decode_tps_proxy_bi1": s1.get("decode_tps_proxy"),
        "prompt_tokens_bi0": s0.get("prompt_tokens"),
        "prompt_tokens_bi1": s1.get("prompt_tokens"),
    }
    Path(args.out).write_text(json.dumps(out, indent=2))

    # pretty print
    print("=" * 78)
    print("PR #759  BATCH-INVARIANCE TAX  per-kernel-family ledger (ms / output tok)")
    print("=" * 78)
    print(f"{'family':<22}{'BI0':>10}{'BI1':>10}{'added':>10}{'share':>9}")
    for r in ledger:
        print(f"{r['family']:<22}{r['bi0_ms_per_tok']:>10.4f}"
              f"{r['bi1_ms_per_tok']:>10.4f}{r['added_ms_per_tok']:>10.4f}"
              f"{r['share_of_total_added']*100:>8.1f}%")
    print("-" * 78)
    print(f"{'TOTAL device':<22}{tot0:>10.4f}{tot1:>10.4f}{tot1-tot0:>10.4f}")
    print(f"profiled device BI tax pct = {out['profiled_device_bi_tax_pct']}")
    print(f"TOP family = {out['bi_tax_top_op_family']}  "
          f"share = {out['bi_tax_top_op_share']*100:.1f}%")
    print("-" * 78)
    print("GEMM subsplit control (int4 Marlin body NOT BI-overridden -> should cancel):")
    print(f"  int4 body ms: BI0={sub0['gemm_int4_body_ms']:.2f} "
          f"BI1={sub1['gemm_int4_body_ms']:.2f}  added/tok={gemm_int4_added_pt:+.5f}")
    print(f"  bf16 GEMM ms: BI0={sub0['gemm_bf16_ms']:.2f} "
          f"BI1={sub1['gemm_bf16_ms']:.2f}  added/tok={gemm_bf16_added_pt:+.5f}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
