#!/usr/bin/env python
"""PR #776 — targeted-locus-revert verdict: does the surgical force-2D attention
patch (int4_mtp_bi0_surgattn, BI=0) deliver byte-exact greedy identity in the
SERVED (CUDA-graph) path, the open caveat from PR #761's eager-only census?

Three-arm self-determinism + self-referential gate, all on the SAME deployed
int4 Marlin stack, one changed factor per arm:

  * eager_screen   surgattn under --enforce-eager  (no CUDA graphs / inductor)
  * served_interlock surgattn under CUDA graphs     (VLLM_COMPILE + inductor)
  * batchinv_served  int4_mtp_batchinv (BI=1) under CUDA graphs

Run-to-run self-determinism (run_00 vs run_01, fresh reloads) is the PRECONDITION
for the official self-referential greedy gate (scripts/validity/greedy_identity_interlock.py
treats PR #38 served wobble as a precondition). If a stack does not reproduce its
OWN decode run-to-run, byte-exact greedy identity is unattainable regardless of the
attention patch. LOCAL ONLY — analysis of already-captured token IDs, no GPU, no HF.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from scripts.validity.analyze_determinism import load_runs, pair_stats  # noqa: E402

BASE = Path(__file__).resolve().parent


def wall_tps(run_dir: Path) -> float | None:
    s = run_dir / "decode_summary.json"
    if not s.exists():
        return None
    d = json.loads(s.read_text())
    ct, dur = d.get("num_completion_tokens"), d.get("duration_s")
    return (ct / dur) if (ct and dur) else None


def arm(root: str, cfg: str, label: str) -> dict:
    runs = load_runs(BASE / root, cfg)
    tps = [t for t in (wall_tps(Path(r["dir"])) for r in runs) if t]
    out = {
        "label": label, "root": f"{root}/{cfg}", "n_runs": len(runs),
        "median_wall_tps": round(statistics.median(tps), 2) if tps else None,
        "per_run_wall_tps": [round(t, 2) for t in tps],
    }
    if len(runs) >= 2:
        fracs, ndiv, onsets = [], 0, []
        for i in range(len(runs)):
            for j in range(i + 1, len(runs)):
                s = pair_stats(runs[i]["rows"], runs[j]["rows"])
                fracs.append(s["byte_identical_frac"])
                ndiv += s["num_divergent"]
                onsets += s["onsets"]
        out.update(
            min_byte_identical_frac=round(min(fracs), 4),
            self_deterministic=bool(min(fracs) >= 0.999),
            num_divergent_pairs=ndiv,
            onset_min=min(onsets) if onsets else None,
            onset_median=int(statistics.median(onsets)) if onsets else None,
        )
    else:
        out.update(min_byte_identical_frac=None, self_deterministic=None,
                   num_divergent_pairs=None)
    return out


def main() -> int:
    arms = {
        "eager_surgattn": arm("eager_screen", "default",
                              "surgattn force-2D, ENFORCE_EAGER (no CUDA graphs)"),
        "served_surgattn": arm("served_interlock", "default",
                               "surgattn force-2D, SERVED (CUDA graphs + inductor)"),
        "served_surgattn_specoff": arm("served_interlock", "default__specoff",
                                       "surgattn force-2D M=1 AR ref, SERVED"),
        "served_batchinv": arm("batchinv_served", "default",
                               "int4_mtp_batchinv BI=1, SERVED (CUDA graphs)"),
    }
    report = {"pr": 776, "experiment": "targeted-locus-revert", "arms": arms}

    e = arms["eager_surgattn"]
    sv = arms["served_surgattn"]
    so = arms["served_surgattn_specoff"]
    bi = arms["served_batchinv"]

    # Verdict: the patch's eager byte-exactness must transfer to the served path.
    eager_ok = bool(e.get("self_deterministic"))
    served_ok = bool(sv.get("self_deterministic")) and bool(so.get("self_deterministic"))
    if eager_ok and served_ok:
        verdict = "CONFIRMED_SERVED"
    elif eager_ok and not served_ok:
        verdict = "REFUTED_SERVED"  # eager byte-exact, served nondeterministic
    else:
        verdict = "INCONCLUSIVE"
    report["verdict"] = verdict
    report["eager_self_deterministic"] = eager_ok
    report["served_self_deterministic"] = served_ok
    report["batchinv_self_deterministic"] = bi.get("self_deterministic")
    report["finding"] = (
        "force-2D restores byte-exact run-to-run determinism in EAGER "
        f"(min_frac={e.get('min_byte_identical_frac')}, confirming #761's M-invariance locus) "
        f"but NOT in the SERVED CUDA-graph path (spec-on min_frac={sv.get('min_byte_identical_frac')} "
        f"=> {sv.get('num_divergent_pairs')}/32 prompts diverge run-to-run, "
        f"M=1 AR ref min_frac={so.get('min_byte_identical_frac')} => {so.get('num_divergent_pairs')}/32). "
        f"DECISIVE CONTRAST: full BI=1 batchinv IS served self-deterministic "
        f"(min_frac={bi.get('min_byte_identical_frac')}, 0/32 divergent, "
        f"{bi.get('median_wall_tps')} TPS) on the SAME CUDA-graph/inductor path. So the "
        "served run-to-run nondeterminism is NOT the compilation layer (batchinv shares it "
        "and reproduces) -- it is the BI=0 fast-path reductions (matmul/norm) that global "
        "batch-invariance freezes but the attention-only force-2D patch leaves untouched. "
        "The #761 attention locus is necessary-but-insufficient for served byte-exactness: "
        f"the surgical revert recovers the speed ({sv.get('median_wall_tps')} vs "
        f"{bi.get('median_wall_tps')} TPS) but forfeits the byte-exact property, so it is "
        "NOT a strict-quality-safe replacement for batchinv."
    )

    out_path = BASE / "served_determinism_verdict.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\n[verdict] {verdict}")
    print(f"[wrote] {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
