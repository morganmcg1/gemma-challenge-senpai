"""Served-ctx-weighted aggregation of the n_seg sweep -> roofline verdict.

Weights the per-shape best-vs-deployed n_seg saving by this submission's real
post-#43 decode ctx distribution (from research/profiling/frontier_decode_postsplitkv
/ ctx_gate_analysis: ctx<256 8.2%, 256-512 43.8%, 512-1024 45.4%, >=1024 2.6%),
and converts the attention-time saving into a TPS-ceiling estimate using the
served composition (attention = 7.6% of GPU-busy, cycle ~= GPU-busy at 99.4%
GPU-bound)."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
d = json.load(open(HERE / "nseg_sweep.json"))
sw = d["sweeps"]

# served ctx distribution -> map to swept ctx points
CTX_WEIGHTS = {128: 0.082, 256: 0.438, 512: 0.454, 1024: 0.026}
ATTN_FRAC_GPU_BUSY = 0.076   # post-#43 served (frontier_decode_postsplitkv)
ATTN_CEILING_TPS = 0.082     # attn->0 ceiling (postsplitkv finding)


def row(lt, ctx, nseg):
    return next(r for r in sw[f"{lt}_ctx{ctx}"]["rows"] if r["n_seg"] == nseg)


per_ctx = {}
agg_dep = agg_best = 0.0
for ctx, w in CTX_WEIGHTS.items():
    s_dep = row("sliding", ctx, 16)["total_us"]
    f_dep = row("full", ctx, 16)["total_us"]
    s_best = sw[f"sliding_ctx{ctx}"]["best_us"]
    f_best = sw[f"full_ctx{ctx}"]["best_us"]
    dep_cyc = 30 * s_dep + 7 * f_dep      # 30 sliding + 7 full layers
    best_cyc = 30 * s_best + 7 * f_best
    per_ctx[ctx] = {
        "weight": w,
        "deployed_us_per_cycle": round(dep_cyc, 1),
        "best_oracle_us_per_cycle": round(best_cyc, 1),
        "saving_frac": round(1 - best_cyc / dep_cyc, 4),
        "best_nseg_sliding": sw[f"sliding_ctx{ctx}"]["best_nseg"],
        "best_nseg_full": sw[f"full_ctx{ctx}"]["best_nseg"],
    }
    agg_dep += w * dep_cyc
    agg_best += w * best_cyc

attn_saving = 1 - agg_best / agg_dep
# ceiling on TPS if we recovered this attention saving (cycle ~= GPU-busy):
tps_uplift = ATTN_FRAC_GPU_BUSY * attn_saving / (1 - ATTN_FRAC_GPU_BUSY * attn_saving)

out = {
    "ctx_weights_source": "frontier_decode_postsplitkv/ctx_gate_analysis served dist",
    "per_ctx": per_ctx,
    "ctx_weighted_deployed_us_per_cycle": round(agg_dep, 1),
    "ctx_weighted_best_oracle_us_per_cycle": round(agg_best, 1),
    "ctx_weighted_attn_time_saving_frac": round(attn_saving, 4),
    "attn_frac_of_gpu_busy": ATTN_FRAC_GPU_BUSY,
    "implied_tps_uplift_oracle_nseg": round(tps_uplift, 5),
    "attn_zero_ceiling_tps": ATTN_CEILING_TPS,
    "note": (
        "oracle = picks optimal n_seg per (layer_type, ctx); the deployed kernel "
        "uses a single global n_seg=16. Realisable served uplift is <= this oracle "
        "and is further bounded by onegraph static CUDA-graph capture (n_seg is a "
        "capture-shape constexpr; per-ctx n_seg breaks the single-graph replay)."
    ),
}
print(json.dumps(out, indent=2))
(HERE / "roofline_summary.json").write_text(json.dumps(out, indent=2))
