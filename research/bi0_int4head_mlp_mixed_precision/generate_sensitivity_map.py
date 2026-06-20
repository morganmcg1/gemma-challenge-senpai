#!/usr/bin/env python
"""PR #810 — assemble the committed per-layer sensitivity map + bytes-vs-PPL Pareto.

Reads recon_sensitivity.json (signal a: weight-recon error) and ppl_sensitivity.json
(signal b: single-layer PPL delta + cumulative configs) and writes sensitivity_map.md.
Pure post-processing, no GPU.
"""
from __future__ import annotations
import json
from pathlib import Path

D = Path("research/bi0_int4head_mlp_mixed_precision")

# exact byte accounting (group_size=32, bf16 scales)
ELEMS_PER_LAYER = 3 * 10240 * 2560          # gate + up + down weights
SCALE_ELEMS_PER_LAYER = 3 * 10240 * 80      # bf16 group scales (gate/up/down each out*ng)
BYTES_PER_ELEM = {4: 0.5, 3: 0.375, 2: 0.25}


def layer_mb(bits: int) -> float:
    w = ELEMS_PER_LAYER * BYTES_PER_ELEM[bits]
    s = SCALE_ELEMS_PER_LAYER * 2
    return (w + s) / 1e6


def main() -> None:
    recon = json.load(open(D / "recon_sensitivity.json"))
    ppl = json.load(open(D / "ppl_sensitivity.json"))

    per_layer = {int(k): v for k, v in recon["per_layer"].items()}
    sweep = ppl["single_layer_sweep"]
    d3 = {s["layer"]: s["delta_ppl"] for s in sweep if s["bits"] == 3}
    d2 = {s["layer"]: s["delta_ppl"] for s in sweep if s["bits"] == 2}
    base = ppl["baseline"]
    cfgs = ppl.get("cumulative_configs", {})

    w4_mb, w3_mb, w2_mb = layer_mb(4), layer_mb(3), layer_mb(2)
    body_w4_gb = 42 * w4_mb / 1e3

    lines = []
    lines.append("# PR #810 — Body-MLP per-layer precision sensitivity map (stark)\n")
    lines.append("LOCAL A10G offline analysis. Base = int4head body "
                 "(`gemma-4-E4B-it-qat-w4a16-ct`, W4A16 g32 sym, 42 layers). "
                 "Reference for recon = dequantized-W4 (`W4deq`, what the base serves).\n")
    lines.append(f"- Offline baseline PPL (all-W4, transformers bf16-dequant, official "
                 f"128-record harness): **{base['full_ppl']:.4f}** "
                 f"(official served int4head base = 2.0029; Δ {base['full_ppl']-2.0029:+.4f} "
                 f"= harness numerics, validates the offline replica).")
    lines.append(f"- Body-MLP weight read @ W4 = **{body_w4_gb:.2f} GB/token** "
                 f"(per layer {w4_mb:.2f} MB; W3 {w3_mb:.2f} MB, W2 {w2_mb:.2f} MB — "
                 f"bf16 g32 scales {(SCALE_ELEMS_PER_LAYER*2)/1e6:.2f} MB/layer are a FIXED "
                 f"overhead, so W3 saves {(1-w3_mb/w4_mb)*100:.0f}% / W2 {(1-w2_mb/w4_mb)*100:.0f}% "
                 f"of layer bytes, below the naive 25%/50%).\n")

    lines.append("## SERVE VIABILITY (Step-2 kill-gate): **FAIL** — sub-4-bit does not "
                 "serve on sm_86 under vLLM 0.22.0.")
    lines.append("- `compressed_tensors_wNa16.py:66-70` raises "
                 "`ValueError: Unsupported num_bits = {3,2}. Supported = [4, 8]` at scheme "
                 "`__init__` (load-time, pre-kernel). Hard load failure, not a dense fallback.")
    lines.append("- `marlin_utils.py:75-83`: Ampere Marlin int types = `{uint4,uint4b8,"
                 "uint8b128}` only (4/8-bit). No uint3/uint2. Machete (sub-4-bit) is sm_90 + "
                 "g∈{-1,64,128} only. ⇒ no W3/W2 kernel on A10G. Step 3 (served TPS/quality) dead.\n")

    # ---- per-layer table ----
    lines.append("## Per-layer sensitivity (signal a: W4deq recon error; signal b: "
                 "single-layer ΔPPL on official harness)\n")
    lines.append("| Layer | recon W3 rel | W3 SQNR dB | recon W2 rel | W2 SQNR dB | "
                 "ΔPPL@W3 | ΔPPL@W2 |")
    lines.append("|------:|-------------:|-----------:|-------------:|-----------:|"
                 "--------:|--------:|")
    for L in range(42):
        pl = per_layer[L]
        lines.append(f"| {L} | {pl['3']['rel_err']:.4f} | {pl['3']['sqnr_db']:.1f} | "
                     f"{pl['2']['rel_err']:.4f} | {pl['2']['sqnr_db']:.1f} | "
                     f"{d3.get(L, float('nan')):+.4f} | {d2.get(L, float('nan')):+.4f} |")

    # robustness summary
    rank3 = sorted(d3, key=lambda L: d3[L])
    lines.append(f"\n- **Most robust (lowest ΔPPL@W3):** "
                 f"{', '.join(f'L{L}({d3[L]:+.4f})' for L in rank3[:6])}")
    lines.append(f"- **Least robust (highest ΔPPL@W3):** "
                 f"{', '.join(f'L{L}({d3[L]:+.4f})' for L in rank3[-6:])}")
    import statistics
    lines.append(f"- ΔPPL@W3 across 42 layers: min {min(d3.values()):+.4f}, "
                 f"max {max(d3.values()):+.4f}, mean {statistics.mean(d3.values()):+.4f}. "
                 f"recon W3 rel_err spread {min(per_layer[L]['3']['rel_err'] for L in range(42)):.3f}"
                 f"–{max(per_layer[L]['3']['rel_err'] for L in range(42)):.3f} "
                 f"(NEARLY UNIFORM — no large robust subset to exploit).\n")

    # ---- Pareto front ----
    lines.append("## Bytes-saved vs PPL Pareto front (cumulative mixed configs, full 128)\n")
    lines.append("| Config | layers down | weight bytes saved/token | full PPL | ΔPPL | "
                 "PPL ≤ 2.42? |")
    lines.append("|--------|------------:|-------------------------:|---------:|-----:|"
                 ":----------:|")
    # all-W4 anchor
    lines.append(f"| all_W4 (base) | 0 | 0 MB | {base['full_ppl']:.4f} | +0.0000 | ✅ |")
    order = ["robust10_W2", "robust10_W3", "robust21_W3", "all_W3"]
    for name in order:
        if name not in cfgs:
            continue
        r = cfgs[name]
        n = r["n_layers_down"]
        bits = 2 if name.endswith("W2") else 3
        saved_mb = n * (w4_mb - (w2_mb if bits == 2 else w3_mb))
        lines.append(f"| {name} | {n} | {saved_mb:.0f} MB | {r['full_ppl']:.4f} | "
                     f"{r['full_ppl']-base['full_ppl']:+.4f} | "
                     f"{'✅' if r['passes_cap_2.42'] else '❌'} |")

    lines.append(f"\n- Body-MLP @ all-W3 would save ~{42*(w4_mb-w3_mb)/1e3:.2f} GB/token "
                 f"({(1-w3_mb/w4_mb)*100:.0f}% of MLP weight bytes) IF it could serve.")
    lines.append("- **Quality budget is wide open** (huge PPL headroom 2.01→2.42); the "
                 "binding constraint is the SERVE kill-gate, not quality.")
    lines.append("- **Reusable guidance for W4-legal levers (wirbel #807 W4A8, fern #808 "
                 "2:4):** sensitivity is near-uniform, so apply uniformly; if forced to spare "
                 f"layers, spare the least-robust {', '.join('L'+str(L) for L in rank3[-4:])}.")

    out = D / "sensitivity_map.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
