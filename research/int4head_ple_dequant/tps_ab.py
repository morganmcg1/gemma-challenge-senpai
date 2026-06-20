#!/usr/bin/env python
"""PR #805 Step 3 — decode-TPS A/B: int4head+PLE-dequant vs int4head control.

Both arms measured back-to-back on THIS A10G (same engine/kernels/substrate) so
the delta is attributable only to the PLE-input-gate dequant. Workload = official
128 prompts x 512 tokens, conc=1, CUDA graphs ON. Honest local number =
``analysis.tps.measured_steady_gen_tps`` (vLLM's own whole-run engine meter), the
same seam #788/#797 used. N reps per arm -> per-rep + median + CV + delta%.
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path("/workspace/senpai/target")
sys.path.insert(0, str(ROOT))

from scripts.local_validation import harness, paths, serve_profile  # noqa: E402

REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
ARMS = [
    ("pledequant", ROOT / "submissions" / "int4_mtp_bi0_int4head_pledequant"),
    ("int4head_control", ROOT / "submissions" / "int4_mtp_bi0_int4head"),
]
OUT = ROOT / "research" / "int4head_ple_dequant" / "tps"
OUT.mkdir(parents=True, exist_ok=True)


def cv(vals: list[float]) -> float:
    if len(vals) < 2:
        return float("nan")
    return 100.0 * statistics.pstdev(vals) / statistics.fmean(vals)


def main() -> int:
    for note in paths.prepare_local_gpu_env():
        print(f"[tps] {note}", flush=True)
    server_python = harness.ensure_server_venv(
        harness.load_manifest(ARMS[0][1])["dependencies"]
    )
    print(f"[tps] server_python={server_python}  reps={REPS}", flush=True)

    results: dict[str, list[float]] = {a: [] for a, _ in ARMS}
    eaccept: dict[str, list[float]] = {a: [] for a, _ in ARMS}
    for label, sub in ARMS:
        for rep in range(1, REPS + 1):
            out_dir = OUT / label / f"rep{rep}"
            t0 = time.time()
            print(f"\n##### {label} rep{rep}/{REPS} -> {out_dir} #####", flush=True)
            report = serve_profile.run(
                sub, server_python, out_dir,
                num_prompts=paths.NUM_PROMPTS, output_len=paths.OUTPUT_LEN,
                kernel_window_tokens=256, variants=["frontier"], do_kernel=False,
                wandb_name=None, wandb_group="bi0-int4head-ple-dequant",
            )
            a = report["analysis"]
            tps = a["tps"]["measured_steady_gen_tps"]
            ea = a["e_accept"]
            results[label].append(float(tps))
            eaccept[label].append(float(ea))
            print(f"[tps] {label} rep{rep}: steady_gen_tps={tps:.2f}  "
                  f"E_accept={ea:.3f}  ({time.time()-t0:.0f}s)", flush=True)

    print("\n" + "=" * 64)
    print("DECODE TPS A/B (measured_steady_gen_tps, tok/s)")
    print("=" * 64)
    summary = {}
    for label, _ in ARMS:
        v = results[label]
        med = statistics.median(v) if v else float("nan")
        summary[label] = {
            "reps": v, "median": med, "mean": statistics.fmean(v) if v else float("nan"),
            "cv_pct": cv(v), "e_accept_median": statistics.median(eaccept[label]) if eaccept[label] else float("nan"),
        }
        print(f"  {label:18s} reps={['%.2f'%x for x in v]}  "
              f"median={med:.2f}  mean={summary[label]['mean']:.2f}  CV={summary[label]['cv_pct']:.2f}%  "
              f"E_accept~{summary[label]['e_accept_median']:.3f}")
    pd = summary["pledequant"]["median"]
    ct = summary["int4head_control"]["median"]
    delta_abs = pd - ct
    delta_pct = 100.0 * delta_abs / ct if ct else float("nan")
    print("-" * 64)
    print(f"  delta (pledequant - control) = {delta_abs:+.2f} tok/s  ({delta_pct:+.2f}%)")
    print(f"  PR projection: +5.3% (-> ~270.4 from 256.74); confirm gate >= +4%; cap if < +2%")
    summary["delta_abs_tps"] = delta_abs
    summary["delta_pct"] = delta_pct
    (OUT / "tps_ab_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[tps] summary -> {OUT / 'tps_ab_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
