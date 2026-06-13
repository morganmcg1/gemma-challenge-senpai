"""Diff the post-#43 (split-KV ON) decode composition against the #30 baseline.

Reads the two serve_profile.analyze() JSONs and emits:
  * the de-duped decode-cycle composition as % of GPU-busy, % of cycle-wall, and
    absolute us/step — including the inter-op gap / CUDA-graph-launch block
    (cycle_wall - GPU-busy), which the #30 table folded into the cycle row;
  * the per-block shift (delta) vs #30, the named new #2 block;
  * next-lever TPS projections on the local steady-state cost model.

LOCAL profiling diff only. No serving/PPL/greedy surface touched.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

BASE = Path("research/profiling/frontier_decode/frontier_decode_profile.json")          # #30 pre-split-KV
NEW = Path("research/profiling/frontier_decode_postsplitkv/frontier_decode_profile.json")  # post-#43

# human label, analyze() component key
COMPONENTS = [
    ("verify body int4-Marlin GEMM", "verify_body_int4_gemm"),
    ("drafter forward (K=7 MTP)", "drafter_forward"),
    ("verify attention (fa2sw)", "verify_attention_fa2sw"),
    ("verify norm/elementwise", "verify_norm_elementwise"),
    ("sampling", "sampling"),
    ("verify lm_head12k GEMV", "verify_lmhead12k_gemm"),
]


def blocks(a: dict) -> dict:
    """Return {label: {ms, pct_busy, pct_cycle}} incl. the inter-op gap block."""
    c = a["cycle"]
    gpu_busy = c["gpu_busy_ms"]
    cycle_wall = c["cycle_wall_ms"]
    host_overhead = c["host_overhead_ms"]
    comp = a["gpu_busy_composition_frac"]
    out = {}
    for label, key in COMPONENTS:
        frac = comp.get(key, 0.0)
        ms = frac * gpu_busy
        out[label] = {
            "ms": ms,
            "pct_busy": 100.0 * frac,
            "pct_cycle": 100.0 * ms / cycle_wall if cycle_wall else 0.0,
        }
    # 7th block: inter-op gaps / CUDA-graph launch + host scheduling (not GPU-busy)
    out["inter-op gap / CUDA-graph launch"] = {
        "ms": host_overhead,
        "pct_busy": float("nan"),  # not part of GPU-busy by definition
        "pct_cycle": 100.0 * host_overhead / cycle_wall if cycle_wall else 0.0,
    }
    return out


def cyc(a: dict) -> dict:
    return a["cycle"]


def main() -> int:
    if not NEW.exists():
        print(f"post-#43 JSON not found yet: {NEW}", file=sys.stderr)
        return 1
    a30 = json.loads(BASE.read_text())["analysis"]
    a43 = json.loads(NEW.read_text())["analysis"]
    b30, b43 = blocks(a30), blocks(a43)
    c30, c43 = cyc(a30), cyc(a43)

    print("## Steady-state spec-decode cycle (p50)\n")
    print("| quantity | #30 pre-splitKV | post-#43 splitKV | delta |")
    print("|---|--:|--:|--:|")
    for label, key in [
        ("drafter forward GPU (ms)", "drafter_gpu_ms"),
        ("verify forward GPU (ms)", "verify_gpu_ms"),
        ("GPU-busy / cycle (ms)", "gpu_busy_ms"),
        ("inter-op gap / host overhead (ms)", "host_overhead_ms"),
        ("cycle wall (ms)", "cycle_wall_ms"),
    ]:
        v30, v43 = c30[key], c43[key]
        print(f"| {label} | {v30:.3f} | {v43:.3f} | {v43 - v30:+.3f} |")
    print(f"| GPU-busy share of wall | {100*c30['gpu_busy_share_of_wall']:.1f}% "
          f"| {100*c43['gpu_busy_share_of_wall']:.1f}% | "
          f"{100*(c43['gpu_busy_share_of_wall']-c30['gpu_busy_share_of_wall']):+.1f}pp |")
    print(f"| E_accept (tokens/cycle) | {a30['e_accept']:.3f} | {a43['e_accept']:.3f} "
          f"| {a43['e_accept']-a30['e_accept']:+.3f} |")
    print(f"| measured steady gen TPS (local) | {a30['tps']['measured_steady_gen_tps']:.1f} "
          f"| {a43['tps']['measured_steady_gen_tps']:.1f} | "
          f"{a43['tps']['measured_steady_gen_tps']-a30['tps']['measured_steady_gen_tps']:+.1f} |")

    print("\n## De-duped decode composition: % of GPU-busy (decode GPU time) + us/step\n")
    print("| block | #30 % | post-#43 % | delta pp | #30 us | post-#43 us |")
    print("|---|--:|--:|--:|--:|--:|")
    order = sorted([l for l, _ in COMPONENTS], key=lambda l: -b43[l]["ms"])
    for label in order:
        d = b43[label]["pct_busy"] - b30[label]["pct_busy"]
        print(f"| {label} | {b30[label]['pct_busy']:.1f}% | {b43[label]['pct_busy']:.1f}% "
              f"| {d:+.1f} | {1000*b30[label]['ms']:.0f} | {1000*b43[label]['ms']:.0f} |")
    # 7th block (PR-requested): inter-op gap / CUDA-graph launch / host scheduling.
    # = cycle_wall - GPU-busy. NOT GPU time, so reported as absolute us/step.
    io30 = 1000.0 * (c30["cycle_wall_ms"] - c30["gpu_busy_ms"])
    io43 = 1000.0 * (c43["cycle_wall_ms"] - c43["gpu_busy_ms"])
    print(f"\ninter-op gap / CUDA-graph launch / host (us/step, NOT GPU time): "
          f"#30 {io30:+.0f} -> post-#43 {io43:+.0f}")
    print(f"  (gpu_busy_share_of_wall_raw: #30 {100*c30['gpu_busy_share_of_wall_raw']:.1f}% "
          f"-> post-#43 {100*c43['gpu_busy_share_of_wall_raw']:.1f}%; "
          f">=100% => host fully overlapped behind async GPU)")

    # New #2 block (after body GEMM), by absolute ms among GPU-busy components.
    gpu_only = sorted([l for l, _ in COMPONENTS], key=lambda l: -b43[l]["ms"])
    print(f"\n## Block ranking (post-#43, GPU-busy, by us/step)\n")
    for i, label in enumerate(gpu_only, 1):
        print(f"  #{i} {label}: {1000*b43[label]['ms']:.0f} us/step ({b43[label]['pct_busy']:.1f}% busy)")
    new_no2 = gpu_only[1]
    print(f"\n=> NEW #2 block: {new_no2} ({b43[new_no2]['pct_busy']:.1f}% of GPU-busy, "
          f"{1000*b43[new_no2]['ms']:.0f} us/step)")
    print(f"   attention collapsed: {b30['verify attention (fa2sw)']['pct_busy']:.1f}% "
          f"-> {b43['verify attention (fa2sw)']['pct_busy']:.1f}% of GPU-busy")

    # ---- Next-lever projections on the local steady cost model ----
    # cycle ~= GPU-busy (decode is GPU-bound); TPS scales as old_cycle/new_cycle.
    cyc_ms = c43["cycle_wall_ms"]
    steady = a43["tps"]["measured_steady_gen_tps"]
    drafter_ms = c43["drafter_gpu_ms"]
    body_ms = b43["verify body int4-Marlin GEMM"]["ms"]
    attn_ms = b43["verify attention (fa2sw)"]["ms"]
    e_acc = a43["e_accept"]
    print("\n## Next-lever TPS projections (local steady cost model; cycle ~ GPU-busy)\n")
    print(f"baseline post-#43: cycle_wall={cyc_ms:.3f} ms, E_accept={e_acc:.3f}, "
          f"local steady TPS={steady:.1f}")

    def proj(name: str, saved_ms: float):
        new_cyc = cyc_ms - saved_ms
        gain = cyc_ms / new_cyc - 1.0
        print(f"  {name}: -{saved_ms:.3f} ms -> cycle {new_cyc:.3f} ms, "
              f"+{100*gain:.1f}% TPS (local {steady:.0f}->{steady*(1+gain):.0f})")

    proj("stark #47 W8A8 drafter, -30% drafter", 0.30 * drafter_ms)
    proj("stark #47 W8A8 drafter, -40% drafter", 0.40 * drafter_ms)
    proj("body GEMM -15% (lower-bit/sparse body)", 0.15 * body_ms)
    proj("kill ALL remaining attention (-100%)", attn_ms)
    # Acceptance lever (multiplier on TPS, no per-step cost change):
    for tgt in (4.2, 4.5, 5.0):
        gain = tgt / e_acc - 1.0
        print(f"  acceptance E_accept {e_acc:.2f}->{tgt:.2f} (land #9/fern #34): "
              f"+{100*gain:.1f}% TPS (local {steady:.0f}->{steady*(1+gain):.0f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
