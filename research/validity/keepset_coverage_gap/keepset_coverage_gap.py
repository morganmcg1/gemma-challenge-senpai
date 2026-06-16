#!/usr/bin/env python3
"""Offline keepset coverage-gap analysis: WIDTH vs CALIBRATION vs BAKE (PR #528).

Pure set-membership analysis (no serving). Given the base model's greedy-argmax
"needed token" stream (gen_base_greedy.py -> decode.jsonl) it answers which of the
served ship's two lm_head prunes is responsible for the downstream-quality collapse
and which fix is cheapest.

THREE NESTED HEADS (advisor fern #531, vcacv804):
  * full vocab          262144  -- what the base-int4 262k head (fern #535) could emit
  * baked_16k            16384  -- osoi5-v0-baked physical lm_head=[16384,320].
                                   The 262k->16k bake. The CEILING for any osoi5
                                   keepset: a row that is not baked cannot be emitted.
  * ship_12k             12288  -- osoi5-12k-baked = baked_16k + LM_HEAD_PRUNE
                                   (dixie pck04c-12k). What the served ship emits.
  ship_12k subset baked_16k subset full_vocab (serve.py:_prune_lm_head_rows asserts
  the 12k keepset is a subset of the 16k source keepset).

So every base-argmax "needed" token the ship drops is in exactly one slice:
  * Slice A (16k->12k)  in baked_16k, not in ship_12k  -> FREE-FIXABLE: recompute a
                        same-width 12k keepset over a broad calibration set; the row
                        already exists in the baked head, so recovering it costs no
                        TPS. This is the only slice the keepset controls.
  * Slice B (262k->16k) not in baked_16k               -> BAKE-BOUND: no osoi5 keepset
                        (even the full 16384) can emit it; needs the base-int4 262k
                        head. fern #531 showed the AIME collapse is held here (16k
                        recovers 0%), so the free fix is bounded by the bake.

Outputs (per task + pooled): total coverage gap and its A/B split; the baked_16k
coverage ceiling; answer-token drop split by slice; dropped-token histogram by class
x slice; min_width_to_cover_{99,999} (unconstrained base-int4 forecast, flagged when
it exceeds the 16384 osoi5 ceiling); broad_calib_12k coverage BOTH unconstrained and
baked-constrained (the true osoi5 free-fix); a three-way verdict per task --
BAKE-BOUND vs CALIBRATION(free) vs WIDTH(within-16k).

Method grounding (VocabTrim, arXiv:2506.22694): top-K frequency selection is the
canonical reduced-vocab method; target-model-generated calibration is best; same-
width recalibration on a domain-matched set is the recommended fix before widening K.
Held-out (cross-prompt) + k-fold splits avoid the circular in-sample estimate.

Run (repo .venv; CPU-only):
  .venv/bin/python research/validity/keepset_coverage_gap/keepset_coverage_gap.py --self-test
  .venv/bin/python research/validity/keepset_coverage_gap/keepset_coverage_gap.py \
      --decode  research/validity/keepset_coverage_gap/decode.jsonl \
      --keepset research/validity/keepset_coverage_gap/pck04_keepset_12k.json \
      --baked   research/validity/keepset_coverage_gap/osoi5_baked_keepset_16k.json \
      --out     research/validity/keepset_coverage_gap/results.json \
      --wandb --wandb-group keepset-coverage-gap --wandb-name kanna/keepset-coverage-gap
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
from collections import Counter, defaultdict

MODEL_ID = "google/gemma-4-E4B-it"
TASKS = ["mmlu_pro", "gpqa_diamond", "aime2024"]
SHIP_WIDTH = 12288       # osoi5-12k-baked emittable rows
BAKED_WIDTH = 16384      # osoi5-v0-baked physical lm_head rows (the bake ceiling)
FULL_VOCAB = 262144
# Width grid for the UNCONSTRAINED (base-int4-reachable) coverage curve.
WIDTH_GRID = [2000, 4000, 6000, 8000, 10000, 12288, 16384, 20000,
              24000, 32000, 48000, 64000, 96000, 131072]
# Osoi5-REACHABLE width grid (>=12288, capped at the 16384 baked ceiling).
OSOI5_WIDTH_GRID = [12288, 13000, 14000, 15000, 16384]


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------
def load_keepset(path: str):
    d = json.loads(open(path).read())
    keep_ids = list(d["keep_ids"])
    full_vocab = int(d.get("full_vocab") or d.get("vocab_size") or 0)
    return set(keep_ids), full_vocab, d


def load_decode(path: str):
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


# ---------------------------------------------------------------------------
# Token classification for the dropped-token histogram
# ---------------------------------------------------------------------------
_MATH_PUNCT = set("+-*/=<>()[]{}^_|\\.,;:!?'\"$%&~`@#")
_OPTION_LETTERS = set("ABCDEFGHIJ")


def classify_token(surface: str) -> str:
    s = surface.strip()
    if s == "":
        return "whitespace"
    if s.isdigit():
        return "digit"
    if len(s) == 1 and s in _OPTION_LETTERS:
        return "option_letter"
    if all((c in _MATH_PUNCT) for c in s):
        return "math_punct"
    if s.isascii() and all((c.isalpha() or c in "'-") for c in s):
        return "common_english"
    return "other"


# ---------------------------------------------------------------------------
# Coverage primitives -- the 12k/16k/262k three-set slice is the whole point.
# ---------------------------------------------------------------------------
def slice_of(t: int, ship: set, baked: set) -> str:
    """'keep' | 'sliceA' (16k->12k, free-fixable) | 'sliceB' (below-16k, bake-bound)."""
    if t in ship:
        return "keep"
    if t in baked:
        return "sliceA"
    return "sliceB"


def coverage_slices(positions, ship: set, baked: set):
    """positions: iterable of token ids. Returns a dict of position- and type-weighted
    counts split into keep / sliceA / sliceB, plus dropped-id lists for the histogram."""
    n_pos = 0
    pos_keep = pos_A = pos_B = 0
    types = set()
    dropped_pos_A, dropped_pos_B = [], []
    for t in positions:
        n_pos += 1
        types.add(t)
        s = slice_of(t, ship, baked)
        if s == "keep":
            pos_keep += 1
        elif s == "sliceA":
            pos_A += 1
            dropped_pos_A.append(t)
        else:
            pos_B += 1
            dropped_pos_B.append(t)
    typ_keep = sum(1 for t in types if t in ship)
    typ_A = sum(1 for t in types if (t not in ship and t in baked))
    typ_B = sum(1 for t in types if t not in baked)
    n_types = len(types)

    def _r(x, d):
        return (x / d) if d else float("nan")

    return {
        "n_positions": n_pos,
        "n_types": n_types,
        # position-weighted
        "coverage_gap_rate_pos": _r(pos_A + pos_B, n_pos),     # total gap (vs ship 12k)
        "gap_sliceA_rate_pos": _r(pos_A, n_pos),               # 16k->12k, free-fixable
        "gap_sliceB_rate_pos": _r(pos_B, n_pos),               # below-16k, bake-bound
        "baked16k_coverage_pos": _r(pos_keep + pos_A, n_pos),  # ceiling for any osoi5 keepset
        # type-weighted
        "coverage_gap_rate_type": _r(typ_A + typ_B, n_types),
        "gap_sliceA_rate_type": _r(typ_A, n_types),
        "gap_sliceB_rate_type": _r(typ_B, n_types),
        "baked16k_coverage_type": _r(typ_keep + typ_A, n_types),
        # raw counts
        "n_keep_pos": pos_keep, "n_sliceA_pos": pos_A, "n_sliceB_pos": pos_B,
        "n_sliceA_types": typ_A, "n_sliceB_types": typ_B,
        # for histograms / further analysis
        "_dropped_pos_A": dropped_pos_A,
        "_dropped_pos_B": dropped_pos_B,
    }


def _public(d: dict) -> dict:
    """Strip the heavy underscore-prefixed id lists for JSON output."""
    return {k: v for k, v in d.items() if not k.startswith("_")}


# ---------------------------------------------------------------------------
# Self-test (machinery validity; runs without a decode file)
# ---------------------------------------------------------------------------
def self_test(keepset_path: str, baked_path: str) -> int:
    print("[self-test] START", flush=True)
    ship, full_vocab, meta = load_keepset(keepset_path)
    baked, baked_full, baked_meta = load_keepset(baked_path)

    # (a) sizes match what the served ship loads.
    assert len(ship) == SHIP_WIDTH, f"ship keepset size {len(ship)} != {SHIP_WIDTH}"
    assert len(baked) == BAKED_WIDTH, f"baked keepset size {len(baked)} != {BAKED_WIDTH}"
    assert full_vocab == FULL_VOCAB, f"full_vocab {full_vocab} != {FULL_VOCAB}"
    assert baked_full == FULL_VOCAB, f"baked full_vocab {baked_full} != {FULL_VOCAB}"
    print(f"[self-test] ship K={len(ship)} baked K={len(baked)} vocab={full_vocab} OK")

    # (b) nesting: ship_12k subset baked_16k subset full_vocab (serve.py asserts this).
    assert ship.issubset(baked), (
        f"ship 12k NOT subset of baked 16k -- {len(ship - baked)} tokens outside the "
        f"physical head; the LM_HEAD_PRUNE subset invariant is violated")
    assert max(baked) < FULL_VOCAB and min(baked) >= 0
    headroom = baked - ship
    assert len(headroom) == BAKED_WIDTH - SHIP_WIDTH, (
        f"free-fix headroom {len(headroom)} != {BAKED_WIDTH - SHIP_WIDTH}")
    print(f"[self-test] nesting ship subset baked OK; free-fix headroom={len(headroom)} "
          f"baked rows ({BAKED_WIDTH}-{SHIP_WIDTH}) OK")

    # (c) full head -> 100% coverage; baked head -> a Slice-A drop is recoverable,
    #     a below-16k token is Slice-B.
    full_head = set(range(full_vocab))
    cov = coverage_slices([0, 5, 100, 262143, 50000], full_head, full_head)
    assert cov["coverage_gap_rate_pos"] == 0.0, "full head must have 0 gap"
    print(f"[self-test] full-head coverage gap pos={cov['coverage_gap_rate_pos']} (==0) OK")

    a_tok = next(iter(headroom))            # in baked, not in ship -> Slice A
    b_tok = next(t for t in range(full_vocab) if t not in baked)  # below-16k -> Slice B
    keep_tok = next(iter(ship))
    cov2 = coverage_slices([keep_tok, a_tok, b_tok], ship, baked)
    assert cov2["n_keep_pos"] == 1 and cov2["n_sliceA_pos"] == 1 and cov2["n_sliceB_pos"] == 1, cov2
    assert abs(cov2["gap_sliceA_rate_pos"] - 1 / 3) < 1e-9
    assert abs(cov2["baked16k_coverage_pos"] - 2 / 3) < 1e-9  # keep + sliceA covered by 16k
    print(f"[self-test] 3-token slice split keep/A/B = "
          f"{cov2['n_keep_pos']}/{cov2['n_sliceA_pos']}/{cov2['n_sliceB_pos']} OK")

    # (d) hand-checkable membership via the tokenizer.
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(MODEL_ID)
        d_space = tok.encode(" D", add_special_tokens=False)
        assert all(i in ship for i in d_space), "' D' answer token unexpectedly dropped"
        rare = FULL_VOCAB - 2  # 262142, top of vocab -> out of both heads
        assert rare not in ship and rare not in baked, "sentinel rare token in a head"
        print(f"[self-test] ' D' ids={d_space} in ship; rare id {rare} in neither head OK")
    except ImportError:
        print("[self-test] transformers unavailable -> skipped tokenizer membership check")

    print("[self-test] PASS", flush=True)
    return 0


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------
def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--decode")
    ap.add_argument("--keepset", default=os.path.join(here, "pck04_keepset_12k.json"))
    ap.add_argument("--baked", default=os.path.join(here, "osoi5_baked_keepset_16k.json"))
    ap.add_argument("--out", default=os.path.join(here, "results.json"))
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--holdout-frac", type=float, default=0.5)
    ap.add_argument("--holdout-seed", type=int, default=12345)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-group", default="keepset-coverage-gap")
    ap.add_argument("--wandb-name", default="kanna/keepset-coverage-gap")
    args = ap.parse_args()

    if args.self_test:
        return self_test(args.keepset, args.baked)

    if not args.decode:
        print("[analysis] --decode required (or use --self-test)", file=sys.stderr)
        return 2

    ship, full_vocab, keep_meta = load_keepset(args.keepset)
    baked, baked_full, baked_meta = load_keepset(args.baked)
    assert len(ship) == SHIP_WIDTH, f"ship K={len(ship)} != {SHIP_WIDTH}"
    assert len(baked) == BAKED_WIDTH, f"baked K={len(baked)} != {BAKED_WIDTH}"
    assert ship.issubset(baked), "ship 12k is not a subset of baked 16k"
    recs = load_decode(args.decode)
    print(f"[analysis] {len(recs)} decode records; ship K={len(ship)} baked K={len(baked)} "
          f"vocab={full_vocab}", flush=True)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)

    # ---- assemble per-task + pooled position streams ----
    pos_by_task = defaultdict(list)
    rec_positions = []                    # parallel: (task, id, [ids])
    for r in recs:
        ids = r["completion_token_ids"]
        pos_by_task[r["task"]].extend(ids)
        rec_positions.append((r["task"], r["id"], ids))
    pooled = [t for task in TASKS for t in pos_by_task.get(task, [])]

    # =====================================================================
    # STEP 3: coverage-gap metrics + 16k/12k slice decomposition
    # =====================================================================
    per_task = {}
    dropped_pos_A_all, dropped_pos_B_all = [], []
    for task in TASKS:
        cov = coverage_slices(pos_by_task.get(task, []), ship, baked)
        dropped_pos_A_all.extend(cov["_dropped_pos_A"])
        dropped_pos_B_all.extend(cov["_dropped_pos_B"])
        per_task[task] = _public(cov)
    pooled_cov_full = coverage_slices(pooled, ship, baked)
    pooled_cov = _public(pooled_cov_full)

    # ---- answer_token_drop_rate, split by slice (the sharp secondary set) ----
    # canonical in-context answer token: last token of "ANSWER: <gold>" (letters)
    # or the space-led gold integer tokens (AIME). per-task + pooled, A/B split.
    answer_drop = {}
    pooled_ans_tokens = []
    for task in TASKS:
        atoks = []
        nd_samp = ns_samp = 0
        for r in recs:
            if r["task"] != task:
                continue
            gold = str(r["gold_answer"]).strip()
            if r["gold_kind"] == "letter":
                ctx = tok.encode("ANSWER: " + gold, add_special_tokens=False)
                a = ctx[-1:]
            else:
                a = tok.encode(" " + gold, add_special_tokens=False)
            atoks.extend(a)
            ns_samp += 1
            nd_samp += int(any(t not in ship for t in a))
        pooled_ans_tokens.extend(atoks)
        nA = sum(1 for t in atoks if (t not in ship and t in baked))
        nB = sum(1 for t in atoks if t not in baked)
        n = len(atoks)
        answer_drop[task] = {
            "answer_token_drop_rate": ((nA + nB) / n) if n else float("nan"),
            "answer_token_drop_sliceA": (nA / n) if n else float("nan"),
            "answer_token_drop_sliceB": (nB / n) if n else float("nan"),
            "answer_any_dropped_sample_rate": (nd_samp / ns_samp) if ns_samp else float("nan"),
            "n_answer_tokens": n, "n_samples": ns_samp,
        }
    nA_p = sum(1 for t in pooled_ans_tokens if (t not in ship and t in baked))
    nB_p = sum(1 for t in pooled_ans_tokens if t not in baked)
    np_ans = len(pooled_ans_tokens)
    pooled_answer_drop = {
        "answer_token_drop_rate": ((nA_p + nB_p) / np_ans) if np_ans else float("nan"),
        "answer_token_drop_sliceA": (nA_p / np_ans) if np_ans else float("nan"),
        "answer_token_drop_sliceB": (nB_p / np_ans) if np_ans else float("nan"),
        "n_answer_tokens": np_ans,
    }

    # ---- dropped_token_histogram (pooled), class x slice ----
    def _hist(ids):
        c = Counter()
        for t in ids:
            c[classify_token(tok.decode([t]))] += 1
        return c
    drop_hist_A = _hist(dropped_pos_A_all)   # 16k->12k, free-fixable
    drop_hist_B = _hist(dropped_pos_B_all)   # below-16k, bake-bound
    classes = sorted(set(list(drop_hist_A) + list(drop_hist_B)))
    dropped_token_histogram = {
        cls: {"sliceA_16k_to_12k": drop_hist_A.get(cls, 0),
              "sliceB_below_16k": drop_hist_B.get(cls, 0)}
        for cls in classes
    }
    # top dropped tokens (pooled, both slices) for qualitative read
    top_dropped = Counter(dropped_pos_A_all + dropped_pos_B_all).most_common(40)
    top_dropped_view = [
        {"id": t, "count": c, "surface": tok.decode([t]),
         "class": classify_token(tok.decode([t])),
         "slice": "sliceA_16k_to_12k" if (t not in ship and t in baked) else "sliceB_below_16k"}
        for t, c in top_dropped
    ]

    # =====================================================================
    # STEP 3b: degeneration-robust mechanism sharpeners. Many base completions
    # hit the 2048/3072 length cap (non-terminating repetitive tails inflate
    # position counts), so length-robust reads:
    #   (1) first_divergence_idx -- earliest step the ship leaves the base path
    #       because a needed token is not in ship_12k (any slice).
    #   (1b) first_sliceB_idx -- earliest step a BAKE-bound token is wanted
    #       (the bake alone, keepset-independent).
    #   (2) stop_only pooled slices -- restricted to finish_reason=="stop".
    #   (3) finish_reason_counts.
    # =====================================================================
    first_div_by_task = defaultdict(list)
    first_bdiv_by_task = defaultdict(list)
    n_samp_by_task = Counter()
    n_div_by_task = Counter()
    n_bdiv_by_task = Counter()
    fr_by_task = defaultdict(Counter)
    stop_pos_by_task = defaultdict(list)
    for r in recs:
        task = r["task"]
        ids = r["completion_token_ids"]
        n_samp_by_task[task] += 1
        fr_by_task[task][str(r.get("finish_reason"))] += 1
        fd = next((i for i, t in enumerate(ids) if t not in ship), None)
        if fd is not None:
            n_div_by_task[task] += 1
            first_div_by_task[task].append(fd)
        fb = next((i for i, t in enumerate(ids) if t not in baked), None)
        if fb is not None:
            n_bdiv_by_task[task] += 1
            first_bdiv_by_task[task].append(fb)
        if str(r.get("finish_reason")) == "stop":
            stop_pos_by_task[task].extend(ids)

    def _div_stats(divs, n_samples, n_div):
        return {
            "n_samples": n_samples,
            "n_diverged": n_div,
            "frac_samples_diverged": (n_div / n_samples) if n_samples else float("nan"),
            "median_first_divergence_idx": (statistics.median(divs) if divs else None),
            "mean_first_divergence_idx": (statistics.fmean(divs) if divs else None),
            "frac_diverged_by_idx10": (sum(1 for x in divs if x < 10) / n_samples)
            if n_samples else float("nan"),
            "frac_diverged_by_idx50": (sum(1 for x in divs if x < 50) / n_samples)
            if n_samples else float("nan"),
        }

    first_divergence, first_div_sliceB = {}, {}
    pooled_divs, pooled_bdivs, pooled_nsamp, pooled_ndiv, pooled_nbdiv = [], [], 0, 0, 0
    for task in TASKS:
        divs = first_div_by_task.get(task, [])
        bdivs = first_bdiv_by_task.get(task, [])
        ns = n_samp_by_task.get(task, 0)
        first_divergence[task] = _div_stats(divs, ns, n_div_by_task.get(task, 0))
        first_div_sliceB[task] = _div_stats(bdivs, ns, n_bdiv_by_task.get(task, 0))
        pooled_divs.extend(divs); pooled_bdivs.extend(bdivs)
        pooled_nsamp += ns; pooled_ndiv += n_div_by_task.get(task, 0)
        pooled_nbdiv += n_bdiv_by_task.get(task, 0)
    first_divergence_pooled = _div_stats(pooled_divs, pooled_nsamp, pooled_ndiv)
    first_div_sliceB_pooled = _div_stats(pooled_bdivs, pooled_nsamp, pooled_nbdiv)

    finish_reason_counts = {task: dict(fr_by_task.get(task, Counter())) for task in TASKS}
    fr_pooled = Counter()
    for task in TASKS:
        fr_pooled.update(fr_by_task.get(task, Counter()))
    finish_reason_counts["POOLED"] = dict(fr_pooled)

    stop_pooled = [t for task in TASKS for t in stop_pos_by_task.get(task, [])]
    stop_cov = coverage_slices(stop_pooled, ship, baked) if stop_pooled else None
    n_stop_samples = sum(1 for r in recs if str(r.get("finish_reason")) == "stop")
    stop_only_pooled = {
        **({k: v for k, v in _public(stop_cov).items()} if stop_cov else {}),
        "n_stop_samples": n_stop_samples,
        "n_total_samples": len(recs),
    }

    # =====================================================================
    # STEP 4 + 5: WIDTH vs CALIBRATION vs BAKE via cross-prompt held-out split
    # =====================================================================
    rng = random.Random(args.holdout_seed)
    train_idx, held_idx = set(), set()
    by_task_idx = defaultdict(list)
    for i, (task, _id, _ids) in enumerate(rec_positions):
        by_task_idx[task].append(i)
    for task, idxs in by_task_idx.items():
        idxs = list(idxs)
        rng.shuffle(idxs)
        ncut = int(round(len(idxs) * (1.0 - args.holdout_frac)))
        train_idx.update(idxs[:ncut])
        held_idx.update(idxs[ncut:])

    train_pos = [t for i in train_idx for t in rec_positions[i][2]]
    held_pos = [t for i in held_idx for t in rec_positions[i][2]]
    n_held = len(held_pos)

    # frequency ranking on TRAIN (canonical top-K freq selection).
    train_counter = Counter(train_pos)
    ranked = [tid for tid, _ in sorted(train_counter.items(),
                                       key=lambda kv: (-kv[1], kv[0]))]
    rank_of = {tid: r for r, tid in enumerate(ranked)}
    INF = float("inf")

    # ---- UNCONSTRAINED min-width forecast (base-int4-reachable; ignores the bake) ----
    held_ranks = [rank_of.get(t, INF) for t in held_pos]
    held_unseen = sum(1 for r in held_ranks if r is INF)
    held_unseen_rate = held_unseen / n_held if n_held else float("nan")
    held_ranks_sorted = sorted(r for r in held_ranks if r is not INF)

    def cov_at_K(K):
        return (sum(1 for r in held_ranks if r < K) / n_held) if n_held else float("nan")
    coverage_curve_unconstrained = {K: cov_at_K(K) for K in WIDTH_GRID}

    def min_width_for(target):
        need = math.ceil(target * n_held)
        if need > len(held_ranks_sorted):
            return None
        return held_ranks_sorted[need - 1] + 1
    min_width_99 = min_width_for(0.99)
    min_width_999 = min_width_for(0.999)
    # is the unconstrained forecast even reachable on the osoi5 ship (<=16384)?
    min_width_99_within_osoi5 = (min_width_99 is not None and min_width_99 <= BAKED_WIDTH)

    # in-sample pooled floor (PR step-4 literal ask): narrowest top-K over POOLED
    # needed tokens. Optimistic lower bound on width (ranking fit on the scored
    # positions); brackets ubel #527's K together with the held-out forecast.
    pooled_counter = Counter(pooled)
    pooled_ranked = [tid for tid, _ in sorted(pooled_counter.items(),
                                              key=lambda kv: (-kv[1], kv[0]))]
    pooled_rank_of = {tid: r for r, tid in enumerate(pooled_ranked)}
    pooled_ranks_sorted = sorted(pooled_rank_of[t] for t in pooled)

    def min_width_insample(target):
        if not pooled_ranks_sorted:
            return None
        need = min(math.ceil(target * len(pooled)), len(pooled_ranks_sorted))
        return pooled_ranks_sorted[need - 1] + 1
    min_width_99_insample = min_width_insample(0.99)
    min_width_999_insample = min_width_insample(0.999)

    # ---- BAKED-CONSTRAINED views (osoi5-reachable; the true free-fix) ----
    # baked_16k coverage = hard ceiling for ANY osoi5 keepset (even width 16384).
    baked16k_cov_pos_held = (sum(1 for t in held_pos if t in baked) / n_held
                             if n_held else float("nan"))
    # broad-recalibrated SAME-WIDTH 12k drawn ONLY from baked rows (the realistic
    # free fix: a row not in the baked head has no logit, so it cannot be kept).
    ranked_baked = [t for t in ranked if t in baked]           # train-frequent, baked only
    broad12k_baked = set(ranked_baked[:SHIP_WIDTH])
    broad12k_baked_cov_held = (sum(1 for t in held_pos if t in broad12k_baked) / n_held
                               if n_held else float("nan"))
    # UNCONSTRAINED broad-12k (top-12288 of all needed; would also need a re-bake) --
    # the gap to the baked-constrained number is the bake's cost at 12k width.
    broad12k_uncon = set(ranked[:SHIP_WIDTH])
    broad12k_uncon_cov_held = (sum(1 for t in held_pos if t in broad12k_uncon) / n_held
                               if n_held else float("nan"))
    # narrow (ship) 12k on the SAME held positions (apples-to-apples baseline).
    narrow12k_cov_held = (sum(1 for t in held_pos if t in ship) / n_held
                          if n_held else float("nan"))
    delta_broad_baked_minus_narrow = broad12k_baked_cov_held - narrow12k_cov_held

    # osoi5-reachable coverage curve: keep the top-K most-frequent baked rows,
    # 12288<=K<=16384. Shows how much widening WITHIN the baked head buys.
    coverage_curve_osoi5 = {}
    for K in OSOI5_WIDTH_GRID:
        # for K>=16384 the whole baked head is kept; otherwise top-K frequent baked rows.
        sel = baked if K >= BAKED_WIDTH else set(ranked_baked[:K])
        coverage_curve_osoi5[K] = (sum(1 for t in held_pos if t in sel) / n_held
                                   if n_held else float("nan"))

    # ---- k-fold cross-validated broad recalibration (robust free-fix test) ----
    def kfold_broad(n_folds, seed):
        rng2 = random.Random(seed)
        fold_of = {}
        for task, idxs in by_task_idx.items():
            order = list(idxs)
            rng2.shuffle(order)
            for j, i in enumerate(order):
                fold_of[i] = j % n_folds
        held_ranks_k = []
        broad_baked_hits = broad_uncon_hits = baked_ceiling_hits = n = 0
        for f in range(n_folds):
            tr_counter = Counter()
            for i in range(len(rec_positions)):
                if fold_of[i] != f:
                    tr_counter.update(rec_positions[i][2])
            ranked_f = [tid for tid, _ in sorted(tr_counter.items(),
                                                 key=lambda kv: (-kv[1], kv[0]))]
            rank_f = {tid: r for r, tid in enumerate(ranked_f)}
            broad_baked_f = set([t for t in ranked_f if t in baked][:SHIP_WIDTH])
            broad_uncon_f = set(ranked_f[:SHIP_WIDTH])
            for i in range(len(rec_positions)):
                if fold_of[i] == f:
                    for t in rec_positions[i][2]:
                        n += 1
                        held_ranks_k.append(rank_f.get(t, INF))
                        if t in broad_baked_f:
                            broad_baked_hits += 1
                        if t in broad_uncon_f:
                            broad_uncon_hits += 1
                        if t in baked:
                            baked_ceiling_hits += 1
        seen_sorted = sorted(r for r in held_ranks_k if r is not INF)

        def mw(target):
            need = math.ceil(target * n)
            return seen_sorted[need - 1] + 1 if need <= len(seen_sorted) else None

        return {
            "n_folds": n_folds,
            "broad_baked_12k_coverage_pos": broad_baked_hits / n if n else float("nan"),
            "broad_uncon_12k_coverage_pos": broad_uncon_hits / n if n else float("nan"),
            "baked16k_ceiling_coverage_pos": baked_ceiling_hits / n if n else float("nan"),
            "unseen_token_rate": sum(1 for r in held_ranks_k if r is INF) / n if n else float("nan"),
            "min_width_99": mw(0.99),
            "min_width_999": mw(0.999),
        }
    kfold = kfold_broad(5, args.holdout_seed)

    # ---- THREE-WAY VERDICT (per task + pooled) ----
    # Decision uses position-weighted coverage on the FULL pooled stream for the
    # ceiling (membership, no generalization gap) and the k-fold broad-12k for the
    # achievable free-fix (cross-prompt, calibration-limited):
    #   BAKE-BOUND   : baked_16k itself covers <99% of needed positions -> even the
    #                  full 16k head can't emit what the base wants; no keepset fix,
    #                  needs the base-int4 262k head (fern #535).
    #   CALIBRATION  : baked_16k covers >=99% AND a same-width broad-recalibrated 12k
    #     (FREE)       drawn from baked rows reaches >=99% -> recompute the keepset,
    #                  same width, same TPS, quality restored.
    #   WIDTH        : baked_16k covers >=99% but no 12k subset does -> must widen the
    #     (within 16k)  keepset toward 16384 (costs TPS, but stays on the osoi5 ship).
    def verdict_for(cov_full, baked_ceiling_cov, broad_baked_cov):
        ceil_ok = (baked_ceiling_cov >= 0.99)
        free_ok = (broad_baked_cov >= 0.99)
        if not ceil_ok:
            return "BAKE-BOUND"
        if free_ok:
            return "CALIBRATION(free)"
        return "WIDTH(within-16k)"

    # per-task ceilings + (in-sample) baked-constrained broad-12k for the verdict.
    per_task_verdict = {}
    for task in TASKS:
        tpos = pos_by_task.get(task, [])
        if not tpos:
            per_task_verdict[task] = {"verdict": "n/a"}
            continue
        ceil_cov = sum(1 for t in tpos if t in baked) / len(tpos)
        tcounter = Counter(tpos)
        tranked_baked = [t for t, _ in sorted(tcounter.items(), key=lambda kv: (-kv[1], kv[0]))
                         if t in baked]
        tbroad = set(tranked_baked[:SHIP_WIDTH])
        broad_cov = sum(1 for t in tpos if t in tbroad) / len(tpos)  # in-sample upper bound
        per_task_verdict[task] = {
            "verdict": verdict_for(None, ceil_cov, broad_cov),
            "baked16k_ceiling_coverage_pos": ceil_cov,
            "broad_baked_12k_coverage_pos_insample": broad_cov,
            "narrow_ship_12k_coverage_pos": sum(1 for t in tpos if t in ship) / len(tpos),
            "answer_token_drop_sliceB": answer_drop[task]["answer_token_drop_sliceB"],
        }

    pooled_verdict = verdict_for(None, kfold["baked16k_ceiling_coverage_pos"],
                                 kfold["broad_baked_12k_coverage_pos"])
    verdict_line = (
        f"POOLED {pooled_verdict}: baked-16k ceiling covers "
        f"{kfold['baked16k_ceiling_coverage_pos']*100:.2f}% of needed positions; "
        f"narrow(ship)-12k {narrow12k_cov_held*100:.2f}% -> a SAME-WIDTH broad-recalibrated "
        f"12k (baked-only) {kfold['broad_baked_12k_coverage_pos']*100:.2f}% "
        f"(+{(kfold['broad_baked_12k_coverage_pos']-narrow12k_cov_held)*100:.2f} pts, 5-fold). "
        f"Unconstrained min_width@99%={min_width_99} "
        f"({'within' if min_width_99_within_osoi5 else 'EXCEEDS'} the 16384 osoi5 ceiling -> "
        f"{'recalibrate/widen on osoi5' if min_width_99_within_osoi5 else 'base-int4 262k only'}). "
        f"Per-task: " + ", ".join(f"{t}={per_task_verdict[t]['verdict']}" for t in TASKS) + "."
    )

    # =====================================================================
    # assemble results
    # =====================================================================
    results = {
        "analysis_only": True,
        "official_tps": 0,
        "model": args.model,
        "keepset_path": args.keepset,
        "baked_path": args.baked,
        "ship_K": len(ship),
        "baked_K": len(baked),
        "full_vocab": full_vocab,
        "free_fix_headroom_rows": BAKED_WIDTH - SHIP_WIDTH,
        "keepset_note": keep_meta.get("note"),
        "baked_note": baked_meta.get("note"),
        "n_records": len(recs),
        "n_records_per_task": {t: n_samp_by_task.get(t, 0) for t in TASKS},
        "holdout_frac": args.holdout_frac,
        "holdout_seed": args.holdout_seed,
        # step 3: coverage + slice decomposition
        "per_task_coverage": per_task,
        "pooled_coverage": pooled_cov,
        "answer_token_drop": answer_drop,
        "pooled_answer_token_drop": pooled_answer_drop,
        "dropped_token_histogram": dropped_token_histogram,
        "top_dropped_tokens": top_dropped_view,
        # step 3b: degeneration-robust mechanism sharpeners
        "first_divergence": first_divergence,
        "first_divergence_pooled": first_divergence_pooled,
        "first_divergence_sliceB": first_div_sliceB,
        "first_divergence_sliceB_pooled": first_div_sliceB_pooled,
        "finish_reason_counts": finish_reason_counts,
        "stop_only_pooled": stop_only_pooled,
        # step 4: width forecast (unconstrained = base-int4-reachable)
        "coverage_curve_unconstrained": coverage_curve_unconstrained,
        "coverage_curve_osoi5_reachable": coverage_curve_osoi5,
        "min_width_to_cover_99": min_width_99,
        "min_width_to_cover_999": min_width_999,
        "min_width_to_cover_99_within_osoi5_ceiling": min_width_99_within_osoi5,
        "min_width_to_cover_99_insample": min_width_99_insample,
        "min_width_to_cover_999_insample": min_width_999_insample,
        "heldout_unseen_token_rate": held_unseen_rate,
        "n_train_positions": len(train_pos),
        "n_heldout_positions": n_held,
        "n_train_prompts": len(train_idx),
        "n_heldout_prompts": len(held_idx),
        # step 5: WIDTH vs CALIBRATION vs BAKE (the free-fix test)
        "baked16k_coverage_pos_heldout": baked16k_cov_pos_held,
        "narrow_ship_12k_coverage_pos_heldout": narrow12k_cov_held,
        "broad_calib_12k_baked_coverage_pos_heldout": broad12k_baked_cov_held,
        "broad_calib_12k_unconstrained_coverage_pos_heldout": broad12k_uncon_cov_held,
        "delta_broad_baked_minus_narrow_pos": delta_broad_baked_minus_narrow,
        # step 5b: k-fold cross-validated (robust)
        "kfold_n_folds": kfold["n_folds"],
        "broad_calib_12k_baked_coverage_pos_kfold": kfold["broad_baked_12k_coverage_pos"],
        "broad_calib_12k_unconstrained_coverage_pos_kfold": kfold["broad_uncon_12k_coverage_pos"],
        "baked16k_ceiling_coverage_pos_kfold": kfold["baked16k_ceiling_coverage_pos"],
        "kfold_unseen_token_rate": kfold["unseen_token_rate"],
        "min_width_to_cover_99_kfold": kfold["min_width_99"],
        "min_width_to_cover_999_kfold": kfold["min_width_999"],
        # verdict
        "per_task_verdict": per_task_verdict,
        "pooled_verdict": pooled_verdict,
        "verdict_line": verdict_line,
    }

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    # ---- console summary ----
    print("\n===== KEEPSET COVERAGE-GAP (12k / 16k-baked / 262k) =====")
    print(f"ship K={len(ship)}  baked K={len(baked)} (free-fix headroom "
          f"{BAKED_WIDTH-SHIP_WIDTH} rows)  vocab={full_vocab}")
    print("--- coverage gap vs SHIP 12k, with 16k->12k (free) vs below-16k (bake) split ---")
    print(f"{'task':14} {'gap_pos':>8} {'sliceA':>8} {'sliceB':>8} {'16kceil':>8} "
          f"{'ans_drop':>9} {'ansB':>7} {'n_pos':>9}")
    for task in TASKS:
        c = per_task[task]; a = answer_drop[task]
        print(f"{task:14} {c['coverage_gap_rate_pos']*100:7.2f}% "
              f"{c['gap_sliceA_rate_pos']*100:7.2f}% {c['gap_sliceB_rate_pos']*100:7.2f}% "
              f"{c['baked16k_coverage_pos']*100:7.2f}% {a['answer_token_drop_rate']*100:8.2f}% "
              f"{a['answer_token_drop_sliceB']*100:6.2f}% {c['n_positions']:9d}")
    pc = pooled_cov; pa = pooled_answer_drop
    print(f"{'POOLED':14} {pc['coverage_gap_rate_pos']*100:7.2f}% "
          f"{pc['gap_sliceA_rate_pos']*100:7.2f}% {pc['gap_sliceB_rate_pos']*100:7.2f}% "
          f"{pc['baked16k_coverage_pos']*100:7.2f}% {pa['answer_token_drop_rate']*100:8.2f}% "
          f"{pa['answer_token_drop_sliceB']*100:6.2f}% {pc['n_positions']:9d}")
    print("--- dropped-token histogram (pos-weighted): sliceA (free) | sliceB (bake) ---")
    for cls in classes:
        h = dropped_token_histogram[cls]
        print(f"  {cls:16} A={h['sliceA_16k_to_12k']:8d}  B={h['sliceB_below_16k']:8d}")
    print("--- mechanism sharpeners ---")
    fp = first_divergence_pooled; bp = first_div_sliceB_pooled
    print(f"first-divergence (any drop, pooled): {fp['frac_samples_diverged']*100:.2f}% diverge; "
          f"median@{fp['median_first_divergence_idx']} by idx10={fp['frac_diverged_by_idx10']*100:.1f}%")
    print(f"first-divergence (BAKE only, pooled): {bp['frac_samples_diverged']*100:.2f}% hit a "
          f"below-16k token; median@{bp['median_first_divergence_idx']}")
    print(f"finish_reason (pooled): {finish_reason_counts['POOLED']}")
    if stop_cov:
        print(f"stop-only pooled gap: pos={stop_only_pooled['coverage_gap_rate_pos']*100:.2f}% "
              f"(sliceA={stop_only_pooled['gap_sliceA_rate_pos']*100:.2f}% "
              f"sliceB={stop_only_pooled['gap_sliceB_rate_pos']*100:.2f}%) "
              f"n_stop={n_stop_samples}/{len(recs)}")
    print("--- WIDTH vs CALIBRATION vs BAKE ---")
    print(f"baked-16k ceiling cov : kfold={kfold['baked16k_ceiling_coverage_pos']*100:.2f}%  "
          f"heldout={baked16k_cov_pos_held*100:.2f}%   (max any osoi5 keepset can reach)")
    print(f"narrow(ship)-12k cov  : heldout={narrow12k_cov_held*100:.2f}%")
    print(f"broad-12k (baked-only): kfold={kfold['broad_baked_12k_coverage_pos']*100:.2f}%  "
          f"heldout={broad12k_baked_cov_held*100:.2f}%   <- the FREE fix (same width/TPS)")
    print(f"broad-12k (uncon/base-int4): kfold={kfold['broad_uncon_12k_coverage_pos']*100:.2f}%  "
          f"(gap to baked-only = bake cost at 12k)")
    print(f"min_width@99 (uncon)  : kfold={kfold['min_width_99']}  single={min_width_99}  "
          f"in-sample-floor={min_width_99_insample}  "
          f"[{'<=' if min_width_99_within_osoi5 else '>'} 16384 ceiling]")
    print(f"VERDICT: {verdict_line}")
    print(f"\n[analysis] wrote {args.out}", flush=True)

    if args.wandb:
        log_wandb(args, results, per_task, answer_drop, dropped_token_histogram,
                  coverage_curve_unconstrained, coverage_curve_osoi5, top_dropped_view,
                  per_task_verdict, classes)
    return 0


def log_wandb(args, results, per_task, answer_drop, dropped_token_histogram,
              coverage_curve_unconstrained, coverage_curve_osoi5, top_dropped_view,
              per_task_verdict, classes):
    import wandb

    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai"),
        entity=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
        group=args.wandb_group,
        name=args.wandb_name,
        job_type="analysis",
        config={
            "analysis_only": True, "official_tps": 0, "pr": 528, "model": args.model,
            "ship_K": results["ship_K"], "baked_K": results["baked_K"],
            "full_vocab": results["full_vocab"],
            "free_fix_headroom_rows": results["free_fix_headroom_rows"],
            "holdout_frac": args.holdout_frac, "holdout_seed": args.holdout_seed,
        },
    )
    pc = results["pooled_coverage"]
    flat = {
        "pooled/coverage_gap_rate_pos": pc["coverage_gap_rate_pos"],
        "pooled/gap_sliceA_rate_pos": pc["gap_sliceA_rate_pos"],
        "pooled/gap_sliceB_rate_pos": pc["gap_sliceB_rate_pos"],
        "pooled/baked16k_coverage_pos": pc["baked16k_coverage_pos"],
        "pooled/coverage_gap_rate_type": pc["coverage_gap_rate_type"],
        "pooled/answer_token_drop_rate": results["pooled_answer_token_drop"]["answer_token_drop_rate"],
        "pooled/answer_token_drop_sliceA": results["pooled_answer_token_drop"]["answer_token_drop_sliceA"],
        "pooled/answer_token_drop_sliceB": results["pooled_answer_token_drop"]["answer_token_drop_sliceB"],
        "width/min_width_to_cover_99": results["min_width_to_cover_99"] or -1,
        "width/min_width_to_cover_999": results["min_width_to_cover_999"] or -1,
        "width/min_width_99_within_osoi5_ceiling": int(results["min_width_to_cover_99_within_osoi5_ceiling"]),
        "width/min_width_to_cover_99_insample": results["min_width_to_cover_99_insample"] or -1,
        "width/min_width_to_cover_99_kfold": results["min_width_to_cover_99_kfold"] or -1,
        "width/heldout_unseen_token_rate": results["heldout_unseen_token_rate"],
        "calib/baked16k_ceiling_coverage_pos_kfold": results["baked16k_ceiling_coverage_pos_kfold"],
        "calib/narrow_ship_12k_coverage_pos_heldout": results["narrow_ship_12k_coverage_pos_heldout"],
        "calib/broad_12k_baked_coverage_pos_kfold": results["broad_calib_12k_baked_coverage_pos_kfold"],
        "calib/broad_12k_unconstrained_coverage_pos_kfold": results["broad_calib_12k_unconstrained_coverage_pos_kfold"],
        "calib/broad_12k_baked_coverage_pos_heldout": results["broad_calib_12k_baked_coverage_pos_heldout"],
        "calib/delta_broad_baked_minus_narrow_pos": results["delta_broad_baked_minus_narrow_pos"],
        "calib/kfold_unseen_token_rate": results["kfold_unseen_token_rate"],
        # mechanism sharpeners
        "mech/frac_samples_diverged": results["first_divergence_pooled"]["frac_samples_diverged"],
        "mech/median_first_divergence_idx": results["first_divergence_pooled"]["median_first_divergence_idx"] or -1,
        "mech/frac_diverged_by_idx10": results["first_divergence_pooled"]["frac_diverged_by_idx10"],
        "mech/sliceB_frac_samples_diverged": results["first_divergence_sliceB_pooled"]["frac_samples_diverged"],
        "mech/sliceB_median_first_divergence_idx": results["first_divergence_sliceB_pooled"]["median_first_divergence_idx"] or -1,
        "mech/stop_only_gap_pos": results["stop_only_pooled"].get("coverage_gap_rate_pos", float("nan")),
        "mech/stop_only_gap_sliceB_pos": results["stop_only_pooled"].get("gap_sliceB_rate_pos", float("nan")),
        "mech/n_stop_samples": results["stop_only_pooled"]["n_stop_samples"],
        "pooled_verdict": results["pooled_verdict"],
    }
    for task in TASKS:
        c = per_task[task]; v = per_task_verdict[task]
        flat[f"{task}/coverage_gap_rate_pos"] = c["coverage_gap_rate_pos"]
        flat[f"{task}/gap_sliceA_rate_pos"] = c["gap_sliceA_rate_pos"]
        flat[f"{task}/gap_sliceB_rate_pos"] = c["gap_sliceB_rate_pos"]
        flat[f"{task}/baked16k_coverage_pos"] = c["baked16k_coverage_pos"]
        flat[f"{task}/answer_token_drop_rate"] = answer_drop[task]["answer_token_drop_rate"]
        flat[f"{task}/answer_token_drop_sliceB"] = answer_drop[task]["answer_token_drop_sliceB"]
        flat[f"{task}/n_positions"] = c["n_positions"]
        flat[f"{task}/verdict"] = v.get("verdict")
    wandb.log({k: v for k, v in flat.items()})

    # per-task coverage + slice + verdict table
    t1 = wandb.Table(columns=["task", "coverage_gap_pos", "sliceA_16k_to_12k",
                              "sliceB_below_16k", "baked16k_coverage", "answer_drop",
                              "answer_drop_sliceB", "verdict", "n_positions"])
    for task in TASKS:
        c = per_task[task]; a = answer_drop[task]; v = per_task_verdict[task]
        t1.add_data(task, c["coverage_gap_rate_pos"], c["gap_sliceA_rate_pos"],
                    c["gap_sliceB_rate_pos"], c["baked16k_coverage_pos"],
                    a["answer_token_drop_rate"], a["answer_token_drop_sliceB"],
                    v.get("verdict"), c["n_positions"])
    t1.add_data("POOLED", pc["coverage_gap_rate_pos"], pc["gap_sliceA_rate_pos"],
                pc["gap_sliceB_rate_pos"], pc["baked16k_coverage_pos"],
                results["pooled_answer_token_drop"]["answer_token_drop_rate"],
                results["pooled_answer_token_drop"]["answer_token_drop_sliceB"],
                results["pooled_verdict"], pc["n_positions"])
    wandb.log({"coverage_table": t1})

    t2 = wandb.Table(columns=["class", "sliceA_16k_to_12k", "sliceB_below_16k"])
    for cls in classes:
        h = dropped_token_histogram[cls]
        t2.add_data(cls, h["sliceA_16k_to_12k"], h["sliceB_below_16k"])
    wandb.log({"dropped_token_histogram": t2})

    t3 = wandb.Table(columns=["K", "unconstrained_cov", "osoi5_reachable_cov"])
    allK = sorted(set(list(coverage_curve_unconstrained) + list(coverage_curve_osoi5)))
    for K in allK:
        t3.add_data(K, coverage_curve_unconstrained.get(K), coverage_curve_osoi5.get(K))
    wandb.log({"coverage_curve": t3})

    t4 = wandb.Table(columns=["id", "count", "surface", "class", "slice"])
    for d in top_dropped_view:
        t4.add_data(d["id"], d["count"], d["surface"], d["class"], d["slice"])
    wandb.log({"top_dropped_tokens": t4})

    wandb.summary.update({k: v for k, v in flat.items() if not isinstance(v, str)})
    wandb.summary["pooled_verdict"] = results["pooled_verdict"]
    wandb.summary["verdict_line"] = results["verdict_line"]
    for task in TASKS:
        wandb.summary[f"{task}/verdict"] = per_task_verdict[task].get("verdict")
    print(f"[wandb] run {run.id} ({run.name})", flush=True)
    run.finish()


if __name__ == "__main__":
    raise SystemExit(main())
