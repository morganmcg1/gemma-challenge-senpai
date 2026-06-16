"""PR #523 byte-exactness sweep -- corroborate that EVERY geometry-sweep segment
count stays byte-exact (0/8 straddle flips), so the served TPS sweep in
``run_realization_gap.py`` is a sweep over *quality-safe* configs only.

Drives the packaged ``verify_packaged_patch.py`` M-invariance microbench once per
(T, S) config (the kernel literal ``tiles_per_segment = T`` is baked at install,
so one T per process). All four fixed configs hold coverage = S*T*16 = 4096 keys
(= max_model_len) constant while varying segment granularity:

    (T=16,S=16) (T=8,S=32) (T=4,S=64, packaged) (T=2,S=128)

plus the ``adaptive`` contrast (deployed nseg=16) which is expected to FLIP at the
straddle boundary -- the control that proves the fixed lever is load-bearing.

Each config: straddle bases {256,512,2048} + a control, M=8 verify vs M=1 AR,
int16 byte-equality. ``flips == 0`` at every straddle => byte-exact / M-invariant.

LOCAL only. No serve, no HF job. Run under repo .venv (resolves the serve venv)::

    .venv/bin/python -m research.speed.byteexact_realization_gap.run_microbench_sweep
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.local_validation import harness  # noqa: E402

OUT_DIR = ROOT / "research" / "speed" / "byteexact_realization_gap" / "microbench"
MICROBENCH = ROOT / "research" / "speed" / "byteexact_attn" / "verify_packaged_patch.py"

# (label, mode, fixed_tps, nseg). Fixed configs hold coverage = T*S*16 = 4096.
CONFIGS = [
    ("adaptive_nseg16", "adaptive", 4, 16),    # contrast: deployed adaptive -> expect flips
    ("fixed_T16_S16", "fixed", 16, 16),        # 2 segs @ L=512
    ("fixed_T8_S32", "fixed", 8, 32),          # 4 segs @ L=512
    ("fixed_T4_S64", "fixed", 4, 64),          # 8 segs @ L=512 (packaged)
    ("fixed_T2_S128", "fixed", 2, 128),        # 16 segs @ L=512
]


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base_manifest = harness.load_manifest(
        (ROOT / "submissions" / "fa2sw_strict_byteexact_splitkv399").resolve())
    server_python = harness.ensure_server_venv(base_manifest["dependencies"])
    print(f"[sweep] server_python={server_python}", flush=True)

    rows: dict[str, dict] = {}
    for label, mode, tps, nseg in CONFIGS:
        out_json = OUT_DIR / f"{label}.json"
        cmd = [
            str(server_python), str(MICROBENCH),
            "--mode", mode, "--nseg", str(nseg), "--fixed-tps", str(tps),
            "--out", str(out_json),
        ]
        print(f"\n[sweep] ===== {label} :: mode={mode} T={tps} S={nseg} =====", flush=True)
        env = {"CUDA_VISIBLE_DEVICES": "0"}
        import os
        full_env = os.environ.copy()
        full_env.update(env)
        proc = subprocess.run(cmd, env=full_env, capture_output=True, text=True)
        sys.stdout.write(proc.stdout)
        if proc.returncode != 0:
            sys.stderr.write(proc.stderr[-2000:])
        try:
            res = json.loads(out_json.read_text())
        except Exception as exc:  # noqa: BLE001
            res = {"error": repr(exc), "returncode": proc.returncode}
        rows[label] = {
            "mode": mode, "fixed_tps": tps, "nseg": nseg,
            "coverage_keys": tps * nseg * 16,
            "straddle_flips_total": res.get("straddle_flips_total"),
            "control_flips": res.get("control_flips"),
            "pass": res.get("pass"),
            "rejit_installed": (res.get("rejit") or {}).get("installed"),
            "backend_num_segments": (res.get("rejit") or {}).get("backend_num_segments"),
            "returncode": proc.returncode,
        }

    fixed = {k: v for k, v in rows.items() if v["mode"] == "fixed"}
    all_fixed_byteexact = bool(fixed) and all(
        v["straddle_flips_total"] == 0 and v["control_flips"] == 0 and v["pass"]
        for v in fixed.values()
    )
    adaptive = rows.get("adaptive_nseg16", {})
    adaptive_flips = adaptive.get("straddle_flips_total")
    summary = {
        "pr": 523,
        "all_fixed_geometry_byteexact_0of8": all_fixed_byteexact,
        "adaptive_contrast_straddle_flips": adaptive_flips,
        "adaptive_contrast_demonstrates_lever": bool(adaptive_flips) if adaptive_flips is not None else None,
        "configs": rows,
    }
    summary_path = OUT_DIR / "microbench_sweep_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\n[sweep] ================= BYTE-EXACT SWEEP SUMMARY =================", flush=True)
    for label, v in rows.items():
        print(f"  {label:18s} mode={v['mode']:8s} T={v['fixed_tps']:2d} S={v['nseg']:3d} "
              f"cov={v['coverage_keys']} straddle_flips={v['straddle_flips_total']} "
              f"control_flips={v['control_flips']} pass={v['pass']}", flush=True)
    print(f"  ALL FIXED GEOMETRY BYTE-EXACT (0/8): {all_fixed_byteexact}", flush=True)
    print(f"  adaptive contrast straddle_flips={adaptive_flips} (>0 => fixed lever load-bearing)", flush=True)
    print(f"[sweep] wrote {summary_path}", flush=True)
    return 0 if all_fixed_byteexact else 1


if __name__ == "__main__":
    raise SystemExit(main())
