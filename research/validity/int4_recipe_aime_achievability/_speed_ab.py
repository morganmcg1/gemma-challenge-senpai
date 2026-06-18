#!/usr/bin/env python
"""PR #679 step-3 speed gate: local served wall-TPS for the group-size sweep.

Reuses the committed #649 measure plumbing (serve flags, capture_decode, the
in-session g128->official 126.378 anchor) but measures MY checkpoints under the
"everything-else-fixed" recipe (lm_head stays int4/g128/untied; only the body
group_size changes):

    anchor  g128 = /workspace/gemma_build/int4_g128_lmhead      (the 126.378 rung)
    cell    g64  = /workspace/gemma_build/int4_g64body_lmhead   (g64 body, same head)
    cell    g32  = /workspace/gemma_build/int4_g32body_lmhead   (g32 body, same head)

official_proj_v = 126.378 * wall_tps_v / wall_tps_g128   (per-variant; drift cancels).

BOTH finer grids clear the AIME bar centrally (g64 mean 0.446, g32 0.433), so per
the PR's step 3 we speed-test BOTH -- the verdict-relevant question is whether the
*cheapest* clearing grid (g64, +3% body bytes) still beats 126.378, not just g32.
PPL is captured on every cell (argmax-independent, cross-session-stable) as a
second quality signal alongside the AIME band. ANALYSIS-ONLY, LOCAL, no HF job.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path("/workspace/senpai/target")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.profiler.g32_recipe_speed_gate as G  # noqa: E402
from scripts.local_validation import harness, paths  # noqa: E402
from scripts.profiler.serveconfig_tps_sweep import DEPS, OFFICIAL_ANCHOR_TPS  # noqa: E402

HERE = Path(__file__).resolve().parent

# Point the int4-head finer-grid corners at MY everything-else-fixed checkpoints,
# in THIS process only -- the committed CELLS file is untouched.
G.CELLS["g32_int4head"]["model"] = "/workspace/gemma_build/int4_g32body_lmhead"
G.CELLS["g32_int4head"]["note"] = "PR#679 g32 body + int4 g128 untied head (everything-else-fixed)"
G.CELLS["g64_int4head"] = {
    "model": "/workspace/gemma_build/int4_g64body_lmhead",
    "body": "g64", "head": "int4-untied",
    "note": "PR#679 g64 body + int4 g128 untied head (everything-else-fixed)",
}

ANCHOR = "g128_int4head"
VARIANTS = ["g64_int4head", "g32_int4head"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-prompts", type=int, default=128)
    ap.add_argument("--output-len", type=int, default=512)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--warmups", type=int, default=1)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--out-dir", type=Path,
                    default=HERE / f"_speed_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    args = ap.parse_args()

    for note in paths.prepare_local_gpu_env():
        print(f"[speed] {note}", flush=True)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    server_py = harness.ensure_server_venv(DEPS)
    print(f"[speed] server_python={server_py} out_dir={out_dir}", flush=True)

    plan = [ANCHOR] + VARIANTS  # anchor first so its tps is the in-session ratio base
    records = []
    for c in plan:
        print(f"\n[speed] === measuring {c} ({G.CELLS[c]['model']}) ===", flush=True)
        rec = G.measure_cell(
            c, server_py, out_dir,
            num_prompts=args.num_prompts, output_len=args.output_len, seeds=[1],
            warmups=args.warmups, reps=args.reps, port=args.port,
            do_selfconsist=False, do_ppl=True,
        )
        records.append(rec)
        print(f"  wall_tps={rec.get('wall_tps')} reps={rec.get('rep_wall_tps')} "
              f"ppl={rec.get('ppl')} ready={rec.get('ready_s')}s mem={rec.get('gpu_mem_used_mib')}MiB "
              f"err={rec.get('error')}", flush=True)

    by = {r["name"]: r for r in records}
    anchor_tps = by[ANCHOR].get("wall_tps")
    variants = {}
    any_beats = False
    for c in VARIANTS:
        v_tps = by[c].get("wall_tps")
        official_proj = delta_pct = None
        beats = False
        if anchor_tps and v_tps and anchor_tps == anchor_tps and v_tps == v_tps:
            official_proj = OFFICIAL_ANCHOR_TPS * v_tps / anchor_tps
            delta_pct = 100.0 * (v_tps - anchor_tps) / anchor_tps
            beats = official_proj > OFFICIAL_ANCHOR_TPS
        any_beats = any_beats or beats
        variants[c] = {
            "body": G.CELLS[c]["body"],
            "local_wall_tps": v_tps,
            "delta_pct_vs_anchor_local": delta_pct,
            "official_proj_tps": official_proj,
            "beats_anchor": beats,
            "ppl": by[c].get("ppl"),
        }

    summary = {
        "official_anchor_tps": OFFICIAL_ANCHOR_TPS,
        "anchor_cell": ANCHOR,
        "anchor_local_wall_tps": anchor_tps,
        "anchor_ppl": by[ANCHOR].get("ppl"),
        "variants": variants,
        "any_variant_beats_anchor": any_beats,
        "workload": {"num_prompts": args.num_prompts, "output_len": args.output_len,
                     "reps": args.reps, "warmups": args.warmups},
    }
    (out_dir / "speed_ab_summary.json").write_text(
        json.dumps({"summary": summary, "records": records}, indent=2))
    (HERE / "speed_ab_summary.json").write_text(
        json.dumps({"summary": summary, "records": records}, indent=2))

    print("\n[speed] ===== SUMMARY =====", flush=True)
    print(f"  anchor {ANCHOR} local wall_tps = {anchor_tps}  (official {OFFICIAL_ANCHOR_TPS})", flush=True)
    for c, v in variants.items():
        dp = v["delta_pct_vs_anchor_local"]
        dp_s = f"{dp:+.2f}%" if dp is not None else "n/a"
        print(f"  {c:16} local={v['local_wall_tps']}  ({dp_s} vs anchor)  "
              f"official_proj={v['official_proj_tps']}  beats={v['beats_anchor']}  ppl={v['ppl']}", flush=True)
    print(f"  any_variant_beats_anchor = {any_beats}", flush=True)
    print(f"[speed] -> {out_dir}/speed_ab_summary.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
