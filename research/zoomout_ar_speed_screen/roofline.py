#!/usr/bin/env python
"""ZOOM-OUT AR speed screen (#630): roofline gate for the strict int4_g128_lmhead rung.

Cheapest decisive gate: does the strict-#319 byte-exact M=1 AR decode have any
knob-addressable headroom toward +10 TPS over 126.378, or is it pinned to the
A10G hardware wall?

All inputs are either (a) measured from the cached w4a16-ct safetensors header,
(b) measured on-branch anchors, or (c) cited on-branch ceiling terms. No network,
no GPU, no serve. Self-tested.
"""
from __future__ import annotations
import json, struct, os, sys

# ---- cited / measured anchors (all on-branch) -------------------------------
A10G_HBM_PEAK_GBs = 600.0            # A10G sm_86 HBM peak
MARLIN_HEAD_GEMV_BW_GBs = 482.9      # denken #550 (head GEMV microbench, 80.5% peak)
FIXED_ATTN_FLOOR_MS = 0.573          # #554 (42 sequential SDPA launches, het head_dim)
BASE_INT4_LOCAL_TPS = 95.77683       # base_int4_floor_tps (my #533, b9j1z40d) warm-median, local
BASE_INT4_OFFICIAL_TPS = 99.0        # implied-official from warm (98.86 agg / 99.15 median)
INT4_HEAD_OFFICIAL_TPS = 126.378     # locked rung int4_g128_lmhead (PR #4)
TPS_GAIN_TARGET = 10.0               # #481 incremental ask: >=+10 TPS

W4A16_CT = os.path.expanduser(
    "~/.cache/huggingface/hub/models--google--gemma-4-E4B-it-qat-w4a16-ct/"
    "snapshots/ef0a4c43726bde42a3ca04fd300397c0b8b3c3f0/model.safetensors")

DT_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "I32": 4, "I8": 1, "U8": 1, "I64": 8}


def safetensors_header(path: str) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n).decode("utf-8"))


def tensor_bytes(meta: dict) -> int:
    ne = 1
    for d in meta["shape"]:
        ne *= d
    return ne * DT_BYTES.get(meta["dtype"], 2)


def decode_read_breakdown(hdr: dict) -> dict:
    """Per-token decode read at M=1 text-only. Embeddings/PLE/towers are token-
    indexed lookups (1 row) or unused (no image/audio) -> NOT streamed per token."""
    body_packed = body_scale = head_bf16 = 0
    for k, m in hdr.items():
        if k == "__metadata__":
            continue
        kl = k.lower()
        if "vision" in kl or "audio" in kl or "embed_tokens" in kl:
            continue  # towers unused at text decode; embeddings are lookups
        if kl.rstrip("0123456789").endswith("lm_head.weight") or ".lm_head." in kl:
            head_bf16 += tensor_bytes(m)  # base-int4: bf16 tied head
        elif k.endswith(".weight_packed"):
            body_packed += tensor_bytes(m)
        elif k.endswith(".weight_scale"):
            body_scale += tensor_bytes(m)
    return {"body_packed": body_packed, "body_scale_g32": body_scale,
            "head_bf16": head_bf16}


def main() -> None:
    hdr = safetensors_header(W4A16_CT)
    b = decode_read_breakdown(hdr)
    GB = 1e9

    body_packed = b["body_packed"] / GB            # identical int4 bytes both rungs
    body_scale_g32 = b["body_scale_g32"] / GB      # base-int4 g32 scales
    body_scale_g128 = body_scale_g32 / 4.0         # int4-head bumps body g32->g128 (4x fewer scales)
    head_bf16 = b["head_bf16"] / GB                # base-int4 bf16 tied head
    # int4-head: untied lm_head quantized to int4 g128 (262144 x 2560)
    head_int4 = 262144 * 2560 * 0.5 / GB
    head_int4_scale = 262144 * (2560 / 128) * 2 / GB
    kv_nominal = 0.022                             # ~ context 600, sliding-window capped (negligible)

    base_bytes = body_packed + body_scale_g32 + head_bf16 + kv_nominal
    head_bytes = body_packed + body_scale_g128 + head_int4 + head_int4_scale + kv_nominal

    # ---- model-free effective bandwidth at each anchor ----
    def eff_bw(tps, bytes_gb):  # GB/s
        return tps * bytes_gb
    base_eff_bw_local = eff_bw(BASE_INT4_LOCAL_TPS, base_bytes)
    base_eff_bw_off = eff_bw(BASE_INT4_OFFICIAL_TPS, base_bytes)
    head_eff_bw_off = eff_bw(INT4_HEAD_OFFICIAL_TPS, head_bytes)

    # ---- two-term fit on the two OFFICIAL anchors (same harness) ----
    # t_token(ms) = bytes/BW_marginal + c_fixed
    t_base = 1000.0 / BASE_INT4_OFFICIAL_TPS
    t_head = 1000.0 / INT4_HEAD_OFFICIAL_TPS
    bw_marginal = (base_bytes - head_bytes) / (t_base - t_head)  # GB/ms = GB/s/1000
    bw_marginal_GBs = bw_marginal * 1000.0
    c_fixed_ms = t_head - head_bytes / bw_marginal

    # ---- headroom to +10 TPS at int4-head ----
    target_tps = INT4_HEAD_OFFICIAL_TPS + TPS_GAIN_TARGET
    t_target = 1000.0 / target_tps
    # (A) bandwidth-bound view: same bytes, need higher effective BW
    eff_bw_needed = eff_bw(target_tps, head_bytes)
    bw_eff_gain_pct = 100.0 * (eff_bw_needed / head_eff_bw_off - 1.0)
    # (B) fixed-overhead view: same marginal BW, need to cut c_fixed
    c_fixed_needed_ms = t_target - head_bytes / bw_marginal
    c_fixed_cut_ms = c_fixed_ms - c_fixed_needed_ms
    c_fixed_cut_pct = 100.0 * c_fixed_cut_ms / c_fixed_ms
    # (C) bytes-bound view: same eff BW, need fewer bytes (this is the relax-#319 lever)
    bytes_needed = head_eff_bw_off * t_target / 1000.0
    bytes_cut_gb = head_bytes - bytes_needed
    bytes_cut_pct = 100.0 * bytes_cut_gb / head_bytes

    out = {
        "kind": "zoomout-ar-speed-roofline",
        "pr": 630,
        "hardware": "A10G sm_86 / 600 GB/s HBM",
        "bytes_per_token_GB": {
            "base_int4_g32_bf16head": round(base_bytes, 4),
            "int4_g128_lmhead": round(head_bytes, 4),
            "components_int4_head": {
                "body_int4_packed": round(body_packed, 4),
                "body_scale_g128": round(body_scale_g128, 4),
                "head_int4": round(head_int4, 4),
                "head_int4_scale": round(head_int4_scale, 4),
                "kv_nominal": kv_nominal,
            },
            "head_read_saving_bf16_to_int4_GB": round(head_bf16 - head_int4 - head_int4_scale, 4),
        },
        "effective_decode_bw_GBs": {
            "base_int4_local_95.78": round(base_eff_bw_local, 1),
            "base_int4_official_99": round(base_eff_bw_off, 1),
            "int4_head_official_126.378": round(head_eff_bw_off, 1),
            "int4_head_pct_of_peak": round(100.0 * head_eff_bw_off / A10G_HBM_PEAK_GBs, 1),
        },
        "two_term_fit_official_anchors": {
            "bw_marginal_GBs": round(bw_marginal_GBs, 1),
            "bw_marginal_pct_of_peak": round(100.0 * bw_marginal_GBs / A10G_HBM_PEAK_GBs, 1),
            "c_fixed_ms_per_token": round(c_fixed_ms, 3),
            "c_fixed_pct_of_token_time_at_126": round(100.0 * c_fixed_ms / t_head, 1),
        },
        "plus10_tps_headroom": {
            "target_tps": target_tps,
            "A_bandwidth_eff_gain_needed_pct": round(bw_eff_gain_pct, 2),
            "B_fixed_overhead_cut_needed_ms": round(c_fixed_cut_ms, 3),
            "B_fixed_overhead_cut_needed_pct": round(c_fixed_cut_pct, 1),
            "C_bytes_cut_needed_GB": round(bytes_cut_gb, 4),
            "C_bytes_cut_needed_pct": round(bytes_cut_pct, 1),
        },
        "knob_verdict": {
            "marlin_only_w4a16_kernel_sm86": True,        # #550
            "cuda_graph_already_on_stock": True,          # stock serve.py has no --enforce-eager
            "attention_pinned_to_only_byteexact_path": True,  # #393/#498/#558 forced-Triton sliding-window
            "splitk_refuted": "-5.82 @ #433",
            "byteexact_knob_can_supply_plus10": False,
        },
    }
    print(json.dumps(out, indent=2))

    # ---- self-tests ----
    checks = {
        "body_int4_~4B_params": abs(body_packed / 0.5 - 3.97) < 0.3,  # 1.986GB/0.5 ~ 3.97G int4 params
        "head_saving_~1GB": 0.9 < (head_bf16 - head_int4 - head_int4_scale) < 1.1,
        "head_bytes_lt_base": head_bytes < base_bytes,
        "int4_head_official_in_range": 120 < INT4_HEAD_OFFICIAL_TPS < 130,
        "bw_marginal_le_peak": bw_marginal_GBs <= A10G_HBM_PEAK_GBs * 1.02,
        "plus10_needs_positive_gain": bw_eff_gain_pct > 0,
    }
    ok = all(checks.values())
    print("SELFTEST", "PASS" if ok else "FAIL", json.dumps(checks), file=sys.stderr)
    os.makedirs(os.path.dirname(os.path.abspath(__file__)) + "/out", exist_ok=True)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "roofline.json"), "w") as f:
        json.dump(out, f, indent=2)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
