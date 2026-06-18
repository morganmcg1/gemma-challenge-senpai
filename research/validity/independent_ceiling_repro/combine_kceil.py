#!/usr/bin/env python
"""Combine round-1 (run/) + fresh-server confirmation (run_confirm/) K-sweep reps.

Resolves the K6 fast-band anomaly: round-1 saw spec_k6 wall_tps=172 (fast ~170
band) even though step latency DROPPED from K5->K6 (physically counter-intuitive,
one extra drafter+verify pass should cost MORE). The fresh-server confirmation
re-measures K4/K5/K6 (2 reps each) so we get median-of-4 per K and can check
whether step_ms(K6) < step_ms(K5) reproduces across independent servers (real
kernel effect) or was a single-server thermal/clock outlier.

Pure read-only aggregation -- no servers, no wandb. Prints the combined verdict
inputs; the wandb log is done by clean_room_kceil.py --finalize on run_final/.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROUND1 = HERE / "run" / "records.jsonl"
CONFIRM = HERE / "run_confirm" / "records.jsonl"

STARK_K6 = 155.58
LAND_K6 = 170.16
TOKENS_PER_REP = 128 * 512  # 65536, ignore_eos full-length

# regime thresholds (mirror clean_room_kceil.finalize)
SLOW_HI = STARK_K6 + (LAND_K6 - STARK_K6) * 0.33   # <= -> slow_155
FAST_LO = STARK_K6 + (LAND_K6 - STARK_K6) * 0.67   # >= -> fast_170


def load(path: Path) -> dict[str, dict]:
    seen: dict[str, dict] = {}
    if not path.exists():
        return seen
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            seen[r["name"]] = r
    return seen


def band(k6: float) -> str:
    if k6 <= SLOW_HI:
        return "slow_155"
    if k6 >= FAST_LO:
        return "fast_170"
    return "intermediate"


def step_stats(rec: dict) -> dict | None:
    """acc_len (emitted/step), step_ms, accept_rate from the server /metrics."""
    m = rec.get("metrics", {}) or {}
    drafts = m.get("vllm:spec_decode_num_drafts_total")
    dtok = m.get("vllm:spec_decode_num_draft_tokens_total")
    acc = m.get("vllm:spec_decode_num_accepted_tokens_total")
    reps = [v for v in rec.get("rep_wall_tps", []) if v == v]
    if not (drafts and acc and reps):
        return None
    nreps = len(reps)
    med = statistics.median(reps)
    acc_len = (acc + drafts) / drafts          # accepted + 1 bonus target / step
    steps_per_rep = drafts / nreps
    rep_dur = TOKENS_PER_REP / med
    step_ms = 1000 * rep_dur / steps_per_rep
    accept_rate = acc / dtok if dtok else float("nan")
    return {"acc_len": acc_len, "step_ms": step_ms, "accept_rate": accept_rate,
            "med_tps": med, "nreps": nreps, "drafts": drafts}


def main() -> None:
    r1 = load(ROUND1)
    cf = load(CONFIRM)
    print(f"thresholds: slow<= {SLOW_HI:.2f}  intermediate  fast>= {FAST_LO:.2f}  "
          f"(stark {STARK_K6} / land {LAND_K6})\n")

    print("per-round step-latency dissection (acc_len deterministic at greedy; "
          "step_ms is the timing variable):")
    print(f"  {'arm':6s} {'round':8s} {'med_tps':>8s} {'acc_len':>8s} {'step_ms':>8s} "
          f"{'acc_rate':>8s} {'reps':>4s}")
    for arm in ("spec_k4", "spec_k5", "spec_k6"):
        for tag, src in (("round1", r1), ("confirm", cf)):
            rec = src.get(arm)
            if not rec:
                print(f"  {arm:6s} {tag:8s} {'(pending)':>8s}")
                continue
            ss = step_stats(rec)
            if ss is None:
                reps = [round(v, 2) for v in rec.get("rep_wall_tps", [])]
                print(f"  {arm:6s} {tag:8s} reps={reps} (no metrics)")
                continue
            print(f"  {arm:6s} {tag:8s} {ss['med_tps']:8.2f} {ss['acc_len']:8.3f} "
                  f"{ss['step_ms']:8.3f} {ss['accept_rate']:8.4f} {ss['nreps']:4d}")
    print()

    print("combined median-of-N per K (round1 reps + confirm reps):")
    combined: dict[str, float] = {}
    for arm in ("spec_k4", "spec_k5", "spec_k6"):
        reps: list[float] = []
        for src in (r1, cf):
            rec = src.get(arm)
            if rec:
                reps += [v for v in rec.get("rep_wall_tps", []) if v == v]
        if not reps:
            print(f"  {arm}: (no reps yet)")
            continue
        med = statistics.median(reps)
        combined[arm] = med
        print(f"  {arm}: median={med:.2f}  n={len(reps)}  reps={[round(v,2) for v in sorted(reps)]}  "
              f"min={min(reps):.2f} max={max(reps):.2f} spread={100*(max(reps)-min(reps))/min(reps):.2f}%")
    print()

    if "spec_k6" in combined:
        k6 = combined["spec_k6"]
        b = band(k6)
        print(f"COMBINED K6 = {k6:.2f}  ->  band={b}")
        print(f"  dist->stark155={abs(k6-STARK_K6):.2f}  dist->land170={abs(k6-LAND_K6):.2f}")
        # anomaly resolution
        if "spec_k5" in combined and "spec_k4" in combined:
            k4, k5 = combined["spec_k4"], combined["spec_k5"]
            print(f"  monotonicity: K4={k4:.2f} K5={k5:.2f} K6={k6:.2f}  "
                  f"(K6>K5 by {k6-k5:+.2f}, K5>K4 by {k5-k4:+.2f})")
    # confirm step_ms reproducibility check
    print("\nANOMALY TEST -- does step_ms(K6) < step_ms(K5) reproduce on fresh servers?")
    for arm in ("spec_k5", "spec_k6"):
        vals = []
        for tag, src in (("round1", r1), ("confirm", cf)):
            rec = src.get(arm)
            ss = step_stats(rec) if rec else None
            if ss:
                vals.append((tag, ss["step_ms"]))
        s = "  ".join(f"{t}={v:.2f}ms" for t, v in vals)
        print(f"  {arm}: {s}")


if __name__ == "__main__":
    main()
