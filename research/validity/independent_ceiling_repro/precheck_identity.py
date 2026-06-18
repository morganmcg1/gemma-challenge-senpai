"""Read-only pre-check of the #319 identity block on round-1 data.

Reuses the exact finalize() helpers so the numbers match what --finalize will log.
Writes nothing.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import clean_room_kceil as crk  # noqa: E402

run = HERE / "run"
by = {}
for line in (run / "records.jsonl").read_text().splitlines():
    if line.strip():
        r = json.loads(line)
        by[r["name"]] = r

# strict sha256 break_rate: spec_k6 vs spec_ar_m1 fingerprints
fp_ref = by["spec_ar_m1"]["fingerprint"]
fp_k6 = by["spec_k6"]["fingerprint"]
n = min(len(fp_ref), len(fp_k6))
n_break = sum(1 for a, b in zip(fp_ref, fp_k6) if a != b)
print("STRICT sha256 break (spec_k6 vs spec_ar_m1): n_break=%d / %d  break_rate=%.6f"
      % (n_break, n, n_break / n))

# per-arm run-floor hazard (rep0 vs rep1 within each arm)
print("\nRUN-FLOOR hazard/step (rep0 vs rep1, same config):")
for a in ("g128_ar", "spec_ar_m1", "spec_k4", "spec_k5", "spec_k6"):
    fl = crk._seq_and_hazard(
        crk._read_completions(run / a / "rep0.jsonl"),
        crk._read_completions(run / a / "rep1.jsonl"))
    print("  %-12s seq_break=%s/%s (%.4f)  root_flips=%s/%s  hazard=%s"
          % (a, fl["seq_break"], fl["n"], (fl["seq_break_rate"] or 0),
             fl["root_flips"], fl["at_risk"], fl["hazard"]))

# spec_k6 vs spec_ar_m1 token-level root hazard
tok = crk._seq_and_hazard(
    crk._read_completions(run / "spec_ar_m1" / "rep0.jsonl"),
    crk._read_completions(run / "spec_k6" / "rep0.jsonl"))
print("\nSPEC_K6 vs SPEC_AR_M1 token root hazard: seq_break=%s/%s root_flips=%s/%s hazard=%s"
      % (tok["seq_break"], tok["n"], tok["root_flips"], tok["at_risk"], tok["hazard"]))

# acceptance / fire from spec_k6 metrics
m = by["spec_k6"].get("metrics", {}) or {}
acc = m.get("vllm:spec_decode_num_accepted_tokens_total")
drf = m.get("vllm:spec_decode_num_draft_tokens_total")
print("\nspec_k6 acceptance_rate=%.4f  spec_fire_rate=%s  (acc=%s draft=%s)"
      % ((acc / drf) if (acc and drf) else float("nan"),
         1.0 if (drf and drf > 0) else 0.0, acc, drf))
print("PPL=%s (ref %s)" % (by.get("ppl", {}).get("ppl"), crk.REF_PPL))
