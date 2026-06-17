#!/usr/bin/env python
"""Offline preview of the #576 spec-dec identity verdicts on already-captured decode files.
No server needed — pure analysis of the decode_*.jsonl token-id captures.

Computes, for REF(no-spec) vs {MTP, ngram} and the partial self-det (ref_a vs ref_b):
  * sequence_exact_rate (the #319 gate; via the OFFICIAL greedy_identity verifier)
  * free-running positional token-identity rate (the verifier's total_divergent_tokens basis)
  * matched-state per-step identity rate = 1 - survival hazard from first-divergence onsets
    (the fern #566-comparable teacher-forced metric: positions before onset are matched-state
     agreements; onset is the first matched-state disagreement)
  * cascade factor seq_miss/step_miss under BOTH per-step definitions
"""
from __future__ import annotations
import json, sys, statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "official" / "main_bucket" / "shared_resources"
                       / "gemma_greedy_identity_verifier_flowian-powers"))
import greedy_identity as gi  # noqa: E402


def matched_state_per_step(report) -> dict:
    """Survival-hazard estimate of the matched-state (teacher-forced) per-position flip rate.
    Divergent prompt with onset d: positions 0..d-1 agreed at matched state, d disagreed ->
    d+1 trials, 1 failure. Identical prompt length L: L trials, 0 failures."""
    trials = 0
    failures = 0
    onsets = []
    for p in report.per_prompt:
        if p.identical:
            trials += p.num_compared
        else:
            d = p.first_divergence_index
            if d is None:
                d = p.num_compared
            trials += d + 1
            failures += 1
            onsets.append(d)
    hazard = failures / trials if trials else None
    return {
        "matched_state_trials": trials,
        "matched_state_failures": failures,
        "matched_state_per_step_disagree": hazard,
        "matched_state_per_step_identity_rate": (1.0 - hazard) if hazard is not None else None,
        "onset_min": min(onsets) if onsets else None,
        "onset_median": statistics.median(onsets) if onsets else None,
        "onset_mean": statistics.mean(onsets) if onsets else None,
        "onset_max": max(onsets) if onsets else None,
    }


def verdict(ref_path, cand_path, label):
    rep = gi.compare_files(ref_path, cand_path)
    n = rep.num_prompts_compared
    seq_exact = rep.num_identical / n if n else None
    freerun_pos = (1.0 - rep.total_divergent_tokens / rep.total_tokens_compared
                   if rep.total_tokens_compared else None)
    ms = matched_state_per_step(rep)
    seq_miss = 1.0 - seq_exact if seq_exact is not None else None
    ms_step_miss = ms["matched_state_per_step_disagree"]
    fr_step_miss = (1.0 - freerun_pos) if freerun_pos is not None else None
    cascade_ms = (seq_miss / ms_step_miss) if (seq_miss and ms_step_miss) else None
    cascade_fr = (seq_miss / fr_step_miss) if (seq_miss and fr_step_miss) else None
    print(f"\n=== {label} ===  verdict={rep.verdict}  n={n}")
    print(f"  num_identical / num_divergent     = {rep.num_identical} / {rep.num_divergent}")
    print(f"  SEQUENCE byte-exact rate          = {seq_exact}")
    print(f"  free-run positional identity rate = {freerun_pos}")
    print(f"  MATCHED-STATE per-step identity   = {ms['matched_state_per_step_identity_rate']}  "
          f"(fern#566-comparable; hazard {ms_step_miss}, {ms['matched_state_failures']}/{ms['matched_state_trials']})")
    print(f"  onsets min/median/mean/max        = {ms['onset_min']}/{ms['onset_median']}/"
          f"{ms['onset_mean']}/{ms['onset_max']}")
    print(f"  cascade factor (seq_miss/step_miss): matched-state={cascade_ms}  free-run={cascade_fr}")
    return {"label": label, "verdict": rep.verdict, "n": n, "seq_exact": seq_exact,
            "freerun_pos": freerun_pos, **ms, "cascade_ms": cascade_ms, "cascade_fr": cascade_fr}


def main():
    ref_a = str(HERE / "decode_ref_a.jsonl")
    mtp = str(HERE / "decode_mtp.jsonl")
    ngram = str(HERE / "decode_ngram.jsonl")
    ref_b = str(HERE / "decode_ref_b.jsonl")
    results = []
    results.append(verdict(ref_a, mtp, "REF(no-spec) vs MTP K=7"))
    results.append(verdict(ref_a, ngram, "REF(no-spec) vs ngram"))
    # self-det on the partial ref_b overlap: subset ref_a to ref_b keys via a temp compare.
    # The official verifier needs matching key sets, so compare on the intersection directly.
    ra = gi.load_decode_outputs(ref_a)
    rb = gi.load_decode_outputs(ref_b)
    common = sorted(set(ra) & set(rb))
    seq_exact_sd = sum(1 for k in common if ra[k]["completion_token_ids"] == rb[k]["completion_token_ids"])
    n_sd = len(common)
    sd_onsets = []
    for k in common:
        a = ra[k]["completion_token_ids"]; b = rb[k]["completion_token_ids"]
        nmin = min(len(a), len(b))
        d = next((i for i in range(nmin) if a[i] != b[i]), None)
        if d is not None or len(a) != len(b):
            sd_onsets.append(d if d is not None else nmin)
    print(f"\n=== SELF-DET (ref_a vs ref_b, PARTIAL {n_sd} common) ===")
    print(f"  byte-exact identical               = {seq_exact_sd}/{n_sd}  "
          f"(rate {seq_exact_sd/n_sd if n_sd else None})")
    print(f"  n divergent (chaotic) prompts      = {len(sd_onsets)}")
    print(f"  self-det divergence onsets         = {sorted(sd_onsets)[:20]}")
    print(f"\n[NOTE] ref_b is partial ({n_sd}/128) — chaos floor is PRELIMINARY; finish ref_b for the real number.")
    (HERE / "offline_preview.json").write_text(json.dumps(
        {"results": results, "self_det_partial": {"n_common": n_sd, "byte_exact": seq_exact_sd,
         "n_chaotic": len(sd_onsets), "onsets": sorted(sd_onsets)}}, indent=2, default=str))


if __name__ == "__main__":
    main()
