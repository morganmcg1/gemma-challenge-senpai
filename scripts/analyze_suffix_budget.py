#!/usr/bin/env python3
"""Offline suffix-run token-budget analysis for SAM-Decoding feasibility (PR #10).

SAM-Decoding (arXiv 2411.10666) accepts verbatim token-run proposals from a
GPU-side suffix automaton "for free" (O(1) drafter cost). This script is the
OFFLINE FEASIBILITY GATE: given the model's *actual* greedy decode outputs on
the 128 public bench prompts, it measures how many generated tokens sit in
verbatim suffix runs longer than K -- the budget that bounds the achievable
TPS gain. It runs no model and launches no HF Job; it only reads a cached
``decode_outputs*.jsonl`` capture.

Two budgets are reported, which bracket the true SAM-Decoding free-token budget:

* PRIMARY -- immediate-suffix-repetition m(t) (advisor spec, PR #10):
    m(t) = longest s such that  context[-s:] == generated[t:t+s],
    context = prompt_ids + generated[:t].
  This detects period-s self-repetition at the boundary t (the block of length
  s ending at t reappears starting at t). It is a strict SUBSET of what a full
  suffix automaton matches, so its budget is a LOWER bracket on the true budget.
  The reported ``frac_tokens_in_run_gt_K`` / histogram / verdict use this.

* SECONDARY -- full earlier-occurrence match (true SAM mechanism):
    at position t, longest L such that generated-block seq[t:t+L] occurs earlier
    anywhere in seq = prompt_ids + generated. A full suffix automaton can retrieve
    any earlier occurrence, so this is an UPPER bracket on the realized budget
    (it does not require the suffix-before-t to also match the source context).
  Reported under ``secondary_full_suffix_match`` with an invariant check
  (full >= immediate on every record).

Aggregation: a single greedy, non-overlapping segmentation. Walk t=0..N-1; if
m(t) >= 2 emit a run of length m(t) and advance by m(t), else advance by 1. A
"run" is a maximal verbatim repeat; runs never overlap, so every generated token
is counted at most once. ``frac_tokens_in_run_gt_K`` = (sum of lengths of runs
with length > K) / total_generated_tokens -- exactly the histogram's tail mass,
and exactly the fraction of tokens a SAM-Decoder would get free at threshold K.
The TPS link: if fraction f of tokens are free, single-stream TPS scales by
1/(1-f) (a free token costs no forward pass), so f ~= 0.036 -> +3.7% TPS.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "research/local_validation/vllm_baseline/decode_outputs_128.jsonl"
DEFAULT_OUTPUT = ROOT / "research/local_validation/suffix_budget/suffix_budget_analysis.json"

K_LEVELS = (4, 6, 8, 10)
PRIMARY_K = 8
DEFAULT_MAX_CHECK = 32  # cap on s/L; > any plausible useful run, bounds O() cost
MIN_RUN = 2  # a "run" is a verbatim repeat of length >= 2


# ---------------------------------------------------------------------------
# PRIMARY: immediate-suffix-repetition m(t) (advisor spec)
# ---------------------------------------------------------------------------
def compute_m_values(prompt_ids: list[int], generated_ids: list[int], max_check: int) -> list[int]:
    """m(t) for each t: longest s (<= max_check) with context[-s:] == generated[t:t+s].

    context = prompt_ids + generated[:t]. Equivalent to longest s such that the
    block ending at t equals the block starting at t (period-s repetition).
    """
    m_values: list[int] = []
    context = list(prompt_ids)
    n = len(generated_ids)
    for t in range(n):
        cap = min(len(context), n - t, max_check)
        best = 0
        # descend so the first hit is the longest match
        for s in range(cap, 0, -1):
            if context[-s:] == generated_ids[t:t + s]:
                best = s
                break
        m_values.append(best)
        context.append(generated_ids[t])
    return m_values


def segment_runs(m_values: list[int]) -> list[tuple[int, int]]:
    """Greedy non-overlapping segmentation -> list of (start_t, run_length) for runs >= MIN_RUN.

    Walk positions; if m(t) >= MIN_RUN emit (t, m(t)) and jump by m(t); else step by 1.
    Non-overlapping: every generated token belongs to at most one run.
    """
    runs: list[tuple[int, int]] = []
    t = 0
    n = len(m_values)
    while t < n:
        s = m_values[t]
        if s >= MIN_RUN:
            runs.append((t, s))
            t += s
        else:
            t += 1
    return runs


# ---------------------------------------------------------------------------
# SECONDARY: full earlier-occurrence match via online suffix automaton
# ---------------------------------------------------------------------------
class SuffixAutomaton:
    """Online generalized suffix automaton over an int sequence, tracking firstpos.

    After feeding seq[0:i], the automaton recognizes every substring of seq[0:i].
    ``firstpos[state]`` is the end index (0-based, in seq) of the *first* occurrence
    of any string ending at that state -- used to test "occurs earlier".
    """

    __slots__ = ("next", "link", "length", "firstpos", "last")

    def __init__(self) -> None:
        self.next: list[dict[int, int]] = [dict()]
        self.link: list[int] = [-1]
        self.length: list[int] = [0]
        self.firstpos: list[int] = [-1]
        self.last = 0

    def extend(self, c: int, pos: int) -> None:
        cur = len(self.length)
        self.next.append(dict())
        self.link.append(-1)
        self.length.append(self.length[self.last] + 1)
        self.firstpos.append(pos)
        p = self.last
        while p != -1 and c not in self.next[p]:
            self.next[p][c] = cur
            p = self.link[p]
        if p == -1:
            self.link[cur] = 0
        else:
            q = self.next[p][c]
            if self.length[p] + 1 == self.length[q]:
                self.link[cur] = q
            else:
                clone = len(self.length)
                self.next.append(dict(self.next[q]))
                self.link.append(self.link[q])
                self.length.append(self.length[p] + 1)
                self.firstpos.append(self.firstpos[q])  # clone inherits first occurrence
                while p != -1 and self.next[p].get(c) == q:
                    self.next[p][c] = clone
                    p = self.link[p]
                self.link[q] = clone
                self.link[cur] = clone
        self.last = cur


def full_match_lengths(prompt_ids: list[int], generated_ids: list[int], max_check: int) -> list[int]:
    """L(t) for each generated t: longest L (<= max_check) such that generated[t:t+L]
    occurs *earlier* in seq = prompt_ids + generated[:t] (start index < absolute t).

    Implemented by feeding seq into the automaton incrementally; before consuming the
    generated token at absolute position ``ap`` we greedily match forward from ap using
    the automaton built on seq[0:ap], requiring each matched prefix to have an earlier
    occurrence (firstpos < ap). This is the true SAM-Decoding retrieval (any earlier
    source), giving an UPPER bracket on the realized free-token budget.
    """
    seq = list(prompt_ids) + list(generated_ids)
    P = len(prompt_ids)
    N = len(generated_ids)
    sam = SuffixAutomaton()
    # seed automaton with the prompt so generation can match against it
    for i in range(P):
        sam.extend(seq[i], i)
    l_values: list[int] = []
    for t in range(N):
        ap = P + t  # absolute index of generated token t in seq
        # greedily match seq[ap:ap+L] in the automaton over seq[0:ap]
        state = 0
        matched = 0
        L = 0
        while matched < max_check and ap + matched < len(seq):
            c = seq[ap + matched]
            nxt = sam.next[state].get(c)
            if nxt is None:
                break
            # require an occurrence ending before the current block ends (=> earlier start)
            if sam.firstpos[nxt] >= ap + matched:
                break
            state = nxt
            matched += 1
            L = matched
        l_values.append(L)
        sam.extend(seq[ap], ap)  # now commit this generated token
    return l_values


def realized_sam_lengths(prompt_ids: list[int], generated_ids: list[int], max_check: int) -> list[int]:
    """r(t) for each generated t: the *realized* SAM-Decoding acceptance length.

    This models the true mechanism: at position t the suffix automaton over seq[:ap]
    (ap = absolute index) is at the longest suffix of seq[:ap] that has an EARLIER
    occurrence (ending at p = firstpos). SAM-Decoding proposes that earlier
    occurrence's continuation seq[p+1:]; the realized acceptance is the common-prefix
    length of seq[ap:] and seq[p+1:] (capped). This is tighter than the LPF upper
    bracket (it requires the preceding context to match, not just the forward block)
    and always >= the immediate-repetition m(t) lower bracket.
    """
    seq = list(prompt_ids) + list(generated_ids)
    P = len(prompt_ids)
    N = len(generated_ids)
    sam = SuffixAutomaton()
    mst, mlen = 0, 0  # longest suffix of consumed prefix with an earlier occurrence
    r_values: list[int] = []
    for ap in range(len(seq)):
        if ap >= P:
            # query realized acceptance at this generated position, using state for seq[:ap]
            r = 0
            if mlen > 0:
                p = sam.firstpos[mst]  # earlier end position of the matched suffix
                if 0 <= p < ap - 1:
                    src = p + 1
                    # the draft can only use tokens that already exist in history
                    # (seq[src:ap]); it cannot copy not-yet-generated tokens. Cap r by
                    # source availability (ap-src), target availability, and max_check.
                    cap = min(max_check, len(seq) - ap, ap - src)
                    while r < cap and seq[ap + r] == seq[src + r]:
                        r += 1
            r_values.append(r)
        c = seq[ap]
        sam.extend(c, ap)
        # update (mst, mlen) to longest suffix of seq[:ap+1] with an earlier occurrence
        if not (c in sam.next[mst] and sam.firstpos[sam.next[mst][c]] < ap):
            while mst != -1 and not (c in sam.next[mst] and sam.firstpos[sam.next[mst][c]] < ap):
                mst = sam.link[mst]
            if mst == -1:
                mst, mlen = 0, 0
                continue
            mlen = sam.length[mst] + 1
            mst = sam.next[mst][c]
        else:
            mst = sam.next[mst][c]
            mlen += 1
    return r_values


def greedy_free_tokens(values: list[int], k: int) -> int:
    """Realized free tokens at threshold K: greedy walk, accept run iff length > k."""
    free, _ = greedy_free_tokens_and_runs(values, k)
    return free


def greedy_free_tokens_and_runs(values: list[int], k: int) -> tuple[int, int]:
    """Greedy walk: accept run iff length > k. Returns (free_tokens, num_runs).

    num_runs lets callers apply the step-saving correction: accepting a run of length
    r costs 1 verification forward pass, so decode steps saved = free_tokens - num_runs.
    """
    t = 0
    free = 0
    runs = 0
    n = len(values)
    while t < n:
        s = values[t]
        if s > k:
            free += s
            runs += 1
            t += s
        else:
            t += 1
    return free, runs


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def distribution_of(rec_id: str) -> str:
    return str(rec_id).split("-", 1)[0]


def load_records(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=str(DEFAULT_INPUT), help="decode_outputs jsonl capture")
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT), help="analysis json output path")
    ap.add_argument("--max-check", type=int, default=DEFAULT_MAX_CHECK)
    ap.add_argument("--expect-prompts", type=int, default=128)
    args = ap.parse_args()

    in_path = Path(args.input)
    records = load_records(in_path)
    if not records:
        raise SystemExit(f"no records in {in_path}")

    n_prompts = len(records)
    total_tokens = 0
    run_hist: Counter[int] = Counter()
    max_run = 0

    # gt-K accounting from the single greedy segmentation
    tokens_in_runs_gt: dict[int, int] = {k: 0 for k in K_LEVELS}
    positions_m_gt: dict[int, int] = {k: 0 for k in K_LEVELS}  # sensitivity sidecar
    # prompt vs output sourced for gt-PRIMARY_K runs
    prompt_sourced_tokens = 0
    output_sourced_tokens = 0
    # per-distribution gt-PRIMARY_K
    dist_tokens: dict[str, int] = defaultdict(int)
    dist_gt_tokens: dict[str, int] = defaultdict(int)
    dist_prompts: dict[str, int] = defaultdict(int)

    # causal SAM-Decoding estimate (the decision-relevant budget) + oracle brackets
    realized_free_gt: dict[int, int] = {k: 0 for k in K_LEVELS}
    realized_runs_gt: dict[int, int] = {k: 0 for k in K_LEVELS}  # for step-saving correction
    lpf_free_gt: dict[int, int] = {k: 0 for k in K_LEVELS}
    dist_realized_gt: dict[str, int] = defaultdict(int)  # realized causal >K8 per distribution
    secondary_error: str | None = None

    for rec in records:
        prompt_ids = rec["prompt_token_ids"]
        generated_ids = rec.get("completion_token_ids") or rec.get("token_ids")
        if generated_ids is None:
            raise SystemExit(f"record {rec.get('id')} missing completion_token_ids/token_ids")
        n = len(generated_ids)
        total_tokens += n
        dist = distribution_of(rec.get("id", ""))
        dist_prompts[dist] += 1
        dist_tokens[dist] += n

        m_values = compute_m_values(prompt_ids, generated_ids, args.max_check)

        # sensitivity: per-position density of m(t) > K
        for k in K_LEVELS:
            positions_m_gt[k] += sum(1 for v in m_values if v > k)

        # single greedy segmentation -> runs, histogram, gt-K token mass
        runs = segment_runs(m_values)
        for (start_t, s) in runs:
            run_hist[s] += 1
            if s > max_run:
                max_run = s
            for k in K_LEVELS:
                if s > k:
                    tokens_in_runs_gt[k] += s
            if s > PRIMARY_K:
                dist_gt_tokens[dist] += s
                # prompt-sourced iff the matched block context[-s:] reached into the
                # prompt, i.e. s > start_t (only possible near generation start)
                if s > start_t:
                    prompt_sourced_tokens += s
                else:
                    output_sourced_tokens += s

        # causal SAM-Decoding estimate + LPF oracle bracket (defensive; never breaks primary)
        if secondary_error is None:
            try:
                r_values = realized_sam_lengths(prompt_ids, generated_ids, args.max_check)
                l_values = full_match_lengths(prompt_ids, generated_ids, args.max_check)
                for k in K_LEVELS:
                    free_r, runs_r = greedy_free_tokens_and_runs(r_values, k)
                    realized_free_gt[k] += free_r
                    realized_runs_gt[k] += runs_r
                    lpf_free_gt[k] += greedy_free_tokens(l_values, k)
                free_r8, _ = greedy_free_tokens_and_runs(r_values, PRIMARY_K)
                dist_realized_gt[dist] += free_r8
            except Exception as exc:  # pragma: no cover - guard only
                secondary_error = repr(exc)

    frac_gt = {f"K{k}": (tokens_in_runs_gt[k] / total_tokens if total_tokens else 0.0) for k in K_LEVELS}
    frac_pos = {f"K{k}": (positions_m_gt[k] / total_tokens if total_tokens else 0.0) for k in K_LEVELS}

    gt_primary_total = prompt_sourced_tokens + output_sourced_tokens
    frac_prompt_sourced = (prompt_sourced_tokens / gt_primary_total) if gt_primary_total else 0.0
    frac_output_sourced = (output_sourced_tokens / gt_primary_total) if gt_primary_total else 0.0

    per_distribution = {}
    for dist in sorted(dist_tokens):
        dt = dist_tokens[dist]
        per_distribution[dist] = {
            "n_prompts": dist_prompts[dist],
            "total_tokens": dt,
            "frac_tokens_in_run_gt_K8": (dist_gt_tokens[dist] / dt) if dt else 0.0,
        }

    def verdict_for(value: float) -> str:
        if value > 0.036:
            return "go"
        if value >= 0.02:
            return "borderline"
        return "no-go"

    primary_value = frac_gt[f"K{PRIMARY_K}"]
    verdict = verdict_for(primary_value)

    causal_available = secondary_error is None
    realized_frac = {f"K{k}": (realized_free_gt[k] / total_tokens if total_tokens else 0.0) for k in K_LEVELS}
    # step-saving fraction = (accepted tokens - accepted runs)/N; one verify pass per run is not free
    realized_step_frac = {
        f"K{k}": ((realized_free_gt[k] - realized_runs_gt[k]) / total_tokens if total_tokens else 0.0)
        for k in K_LEVELS
    }
    lpf_frac = {f"K{k}": (lpf_free_gt[k] / total_tokens if total_tokens else 0.0) for k in K_LEVELS}
    causal_value = realized_frac[f"K{PRIMARY_K}"]
    causal_verdict = verdict_for(causal_value)

    causal_sam_estimate: dict[str, Any] = {
        "description": "Realized, CAUSAL SAM-Decoding free-token budget: at each decode step the "
                       "suffix automaton over only-already-generated tokens proposes the "
                       "continuation of the best earlier suffix match; acceptance is the verbatim "
                       "(greedy-safe) common-prefix length vs the actual output. This is the "
                       "physically achievable budget, unlike the advisor-spec immediate m(t) "
                       "which is non-causal (looks ahead) and adjacent-only.",
        "available": causal_available,
    }
    if causal_available:
        causal_sam_estimate["frac_tokens_free_gt_K"] = realized_frac
        causal_sam_estimate["frac_decode_steps_saved_gt_K"] = realized_step_frac
        causal_sam_estimate["primary_k_value"] = causal_value
        causal_sam_estimate["causal_verdict"] = causal_verdict
        causal_sam_estimate["per_distribution_frac_free_gt_K8"] = {
            d: (dist_realized_gt[d] / dist_tokens[d] if dist_tokens[d] else 0.0) for d in sorted(dist_tokens)
        }
        causal_sam_estimate["lpf_forward_oracle_upper_frac_gt_K"] = lpf_frac
        causal_sam_estimate["bracket_note"] = (
            "immediate m(t) (oracle, adjacent-only) and lpf (oracle, forward-only) bound the "
            "intuition; realized causal is the actionable number. ordering is not strict because "
            "the three measure different things (look-ahead vs causal vs context-free)."
        )
    else:
        causal_sam_estimate["error"] = secondary_error

    result = {
        "n_prompts": n_prompts,
        "total_generated_tokens": total_tokens,
        "max_check": args.max_check,
        "input_file": str(in_path),
        "primary_metric_name": f"frac_tokens_in_run_gt_K{PRIMARY_K}",
        "primary_metric_value": primary_value,
        "frac_tokens_in_run_gt_K": frac_gt,
        "frac_positions_m_gt_K": frac_pos,
        "frac_prompt_sourced_of_gt_K8": frac_prompt_sourced,
        "frac_output_sourced_of_gt_K8": frac_output_sourced,
        "run_length_histogram": {str(s): run_hist[s] for s in sorted(run_hist)},
        "max_run_length": max_run,
        "per_distribution": per_distribution,
        "verdict": verdict,
        "causal_sam_estimate": causal_sam_estimate,
        "causal_value_gt_K8": causal_value if causal_available else None,
        "causal_verdict": causal_verdict if causal_available else None,
        "recommendation": (
            "The advisor-spec primary metric (immediate m(t), K>8 = "
            f"{primary_value:.4f}) yields verdict '{verdict}', but m(t) is the WRONG proxy: it is "
            "non-causal (looks ahead) and only detects adjacent period-s repetition, which is rare "
            "in these reasoning outputs. The realized CAUSAL SAM-Decoding budget (K>8 = "
            f"{causal_value:.4f}) yields verdict '{causal_verdict}'. Base the SAM-Decoding "
            "go/no-go on the causal estimate."
        ) if causal_available else "causal estimate unavailable",
        "notes": (
            "frac_tokens_in_run_gt_K is the fraction of generated tokens inside verbatim "
            "self-repetition runs longer than K, from a single greedy non-overlapping "
            "segmentation of m(t) (advisor spec, immediate period-s repetition). It equals "
            "the histogram tail mass and the realized free-token fraction at proposal threshold "
            "K; single-stream TPS would scale ~1/(1-f). WARNING: this advisor-spec m(t) is a poor "
            "proxy -- non-causal (looks ahead at whether the block after t repeats the block "
            "before t) and detects only ADJACENT period-s repetition, which is rare here. The "
            "physically achievable budget is causal_sam_estimate (realized causal SAM-Decoding); "
            "lpf_forward_oracle_upper_frac_gt_K inside it is a loose non-causal upper reference. "
            "frac_positions_m_gt_K (fraction of decode steps where a >K m(t) proposal is "
            "available) is a sensitivity sidecar, not the budget. Top-level 'verdict' applies the "
            "advisor table (>3.6% go, 2-3.6% borderline, <2% no-go) to the spec m(t) frac at K8; "
            "'causal_verdict' applies it to the realized causal frac -- use causal_verdict for the "
            "SAM-Decoding go/no-go decision."
        ),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=False) + "\n")

    print(f"n_prompts={n_prompts} total_generated_tokens={total_tokens}")
    print(f"frac_tokens_in_run_gt_K (m(t) advisor-spec, primary): {frac_gt}")
    print(f"frac_positions_m_gt_K (sensitivity):                  {frac_pos}")
    if causal_available:
        print(f"CAUSAL SAM realized free_gt_K:                        {realized_frac}")
        print(f"  causal decode-steps-saved_gt_K:                    {realized_step_frac}")
        print(f"  lpf forward-oracle upper_gt_K:                     {lpf_frac}")
        print(f"  causal per-distribution free_gt_K8:                {causal_sam_estimate['per_distribution_frac_free_gt_K8']}")
    print(f"prompt_sourced/output_sourced of m(t)>K8: {frac_prompt_sourced:.4f} / {frac_output_sourced:.4f}")
    print(f"max_run_length={max_run}")
    print(f"PRIMARY (m(t)) frac_tokens_gt_k8={primary_value:.5f}  VERDICT={verdict}")
    if causal_available:
        print(f"CAUSAL  (realized) frac_tokens_gt_k8={causal_value:.5f}  VERDICT={causal_verdict}")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
