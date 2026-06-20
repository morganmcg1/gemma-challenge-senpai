#!/usr/bin/env python
"""PR #805 Step 4 — quality gate for int4head+PLE-dequant.

Runs, all WITHIN THIS JOB (no cached cross-job reference — that landmine bit bi0):
  1. gen pledequant  spec-off M=1 AR reference (128 prompts)
  2. gen int4head    spec-off M=1 AR reference (128 prompts)
  3. validate_submission(pledequant, --official-gate): served (spec-ON) decode ->
       PPL, greedy-identity vs pledequant's OWN spec-off ref (isolates speculation),
       128/128 completions, all-4-modalities, official_gate; logs W&B.
  4. cross-checkpoint greedy A/B: pledequant spec-off  vs  int4head spec-off
       (both M=1 AR -> isolates ONLY the PLE-input-gate bf16-cuBLAS vs int4-Marlin
       delta). Reports N/128 divergent, first-token flips, onset loci.
  5. GSM8K (greedy + sampled, N=200, 8-shot) — floor 0.807, int4head ref 0.9150.

Each serve is sequential (single GPU). Honest PPL ref = int4head 2.0029; cap 2.42.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/workspace/senpai/target")
sys.path.insert(0, str(ROOT))
PY = "/tmp/senpai-venvs/20f658587e8a6643/bin/python"

from scripts.local_validation import gen_greedy_reference as gr  # noqa: E402
from scripts.local_validation import greedy_gate  # noqa: E402

PLEDQ = ROOT / "submissions" / "int4_mtp_bi0_int4head_pledequant"
INT4HEAD = ROOT / "submissions" / "int4_mtp_bi0_int4head"
OUT = ROOT / "research" / "int4head_ple_dequant" / "quality"
OUT.mkdir(parents=True, exist_ok=True)
N = 128


def sh(cmd: list[str], log: Path) -> int:
    print(f"\n$ {' '.join(cmd)}\n  -> log {log}", flush=True)
    t0 = time.time()
    with open(log, "w") as f:
        rc = subprocess.run(cmd, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT).returncode
    print(f"  rc={rc} ({time.time()-t0:.0f}s)", flush=True)
    return rc


def stage_refs() -> dict:
    """Generate both within-job spec-off M=1 AR references."""
    out = {}
    for label, sub in (("pledequant", PLEDQ), ("int4head", INT4HEAD)):
        rc = sh([PY, "-m", "scripts.local_validation.gen_greedy_reference",
                 "--mode", "served", "--submission", str(sub), "--spec-off",
                 "--num-prompts", str(N), "--port", "8001"],
                OUT / f"genref_{label}.log")
        path = gr.served_reference_path(gr.reference_key_for_submission(sub))
        out[label] = {"rc": rc, "ref": str(path), "exists": path.exists()}
        print(f"  {label} spec-off ref: {path} exists={path.exists()}", flush=True)
    return out


def stage_validate() -> int:
    return sh([PY, "-m", "scripts.local_validation.validate_submission",
               "--submission", str(PLEDQ), "--num-prompts", str(N),
               "--official-gate", "--out-dir", str(OUT / "validate"),
               "--wandb-name", "ubel/pledequant-validate",
               "--wandb-group", "bi0-int4head-ple-dequant"],
              OUT / "validate.log")


def stage_ab(refs: dict) -> dict:
    """Cross-checkpoint A/B: pledequant spec-off vs int4head spec-off (isolates PLE)."""
    pledq_ref = Path(refs["pledequant"]["ref"])
    int4_ref = Path(refs["int4head"]["ref"])
    if not (pledq_ref.exists() and int4_ref.exists()):
        return {"error": "missing spec-off reference(s)", "pledq": str(pledq_ref), "int4": str(int4_ref)}
    rep = greedy_gate.compare(int4_ref, pledq_ref)  # reference=int4head, candidate=pledequant
    per = [p for p in rep.per_prompt if not p.identical]
    first_tok_flips = sum(1 for p in per if p.first_divergence_index == 0)
    onset = greedy_gate.onset_summary(rep)
    ab = {
        "verdict": rep.verdict,
        "num_prompts_compared": rep.num_prompts_compared,
        "num_identical": rep.num_identical,
        "num_divergent": rep.num_divergent,
        "total_tokens_compared": rep.total_tokens_compared,
        "total_divergent_tokens": rep.total_divergent_tokens,
        "first_token_flips": first_tok_flips,
        "onset": onset,
        "onset_line": greedy_gate.onset_line(onset, 512),
        "divergent_loci": [
            {"key": p.key, "first_divergence_index": p.first_divergence_index}
            for p in per[:20]
        ],
    }
    (OUT / "ab_pledequant_vs_int4head.json").write_text(json.dumps(ab, indent=2))
    print(f"\n[A/B pledequant-specoff vs int4head-specoff] verdict={ab['verdict']} "
          f"divergent={ab['num_divergent']}/{ab['num_prompts_compared']} "
          f"first_token_flips={first_tok_flips}  {ab['onset_line']}", flush=True)
    return ab


def stage_gsm8k() -> dict:
    out = {}
    for regime in ("greedy", "sampled"):
        rc = sh([PY, "research/downstream_quality_gsm8k/gsm8k_eval.py",
                 "--submission", str(PLEDQ), "--server-python", PY,
                 "--label", f"pledequant_{regime}", "--regimes", regime,
                 "--n", "200", "--n-shot", "8", "--concurrency", "32",
                 "--max-num-seqs", "32", "--port", "8000",
                 "--out-dir", str(OUT / "gsm8k")],
                OUT / f"gsm8k_{regime}.log")
        out[regime] = {"rc": rc}
    return out


def main() -> int:
    summary = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    print("===== STAGE 1+2: within-job spec-off references =====", flush=True)
    refs = stage_refs()
    summary["refs"] = refs
    print("\n===== STAGE 3: validate_submission (PPL/greedy/modalities/gate) =====", flush=True)
    summary["validate_rc"] = stage_validate()
    try:
        ev = json.loads((OUT / "validate" / "evidence.json").read_text())
        summary["evidence"] = {k: ev.get(k) for k in (
            "ppl", "completed", "num_prompts", "all_modalities_loaded",
            "greedy_verdict", "greedy_onset", "official_gate")}
    except Exception as e:
        summary["evidence_error"] = str(e)
    print("\n===== STAGE 4: cross-checkpoint greedy A/B =====", flush=True)
    summary["ab"] = stage_ab(refs)
    print("\n===== STAGE 5: GSM8K (greedy + sampled, N=200) =====", flush=True)
    summary["gsm8k"] = stage_gsm8k()
    (OUT / "quality_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[quality] summary -> {OUT / 'quality_summary.json'}", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
