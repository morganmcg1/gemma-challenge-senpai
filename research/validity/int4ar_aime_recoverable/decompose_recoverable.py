"""PR #668 — int4-body AIME deficit: recoverable-set + first-divergence locus.

Problem-level + token-position-level decomposition of the int4-AR vs bf16 greedy
AIME @12288 deficit, reusing the #650 transcripts. Deliberately NOT a layer-level
scan (that lane is owned elsewhere).

Four deliverables (PR #668):
  1. Recoverable set R = {bf16 @12288 CORRECT and int4-AR @12288 WRONG} (+ the
     reverse set int4-right / bf16-wrong). |R| is the raw recoverable gap.
  2. Concentration — are R problems clustered or spread (year / problem-number
     difficulty tier / coarse type)?
  3. First-divergence locus — for each R problem, greedy int4 vs bf16 from the
     identical prompt, first token position where the argmax streams diverge,
     normalized by completion length. EARLY(<25%) = body-wide reasoning drift,
     LATE(>75%) = final-answer / head locus.
  4. Verdict — RECOVERABLE_CONCENTRATED / RECOVERABLE_DIFFUSE / FUNDAMENTAL.

analysis_only: NO HF Job, NO submission, NO served-file change. The live
int4_g128_lmhead @ 126.378 stays untouched. We log analysis_only=true and
official_tps=0 as explicit W&B summary scalars (the no-fire guard is
machine-checkable).

Subcommands
-----------
  recoverable   parse the two #650 .out files -> R + reverse + concentration
                -> recoverable_set.json   (no GPU, no network unless --tag-type)
  harvest       greedy + logprobs token streams for R+reverse from one served
                endpoint -> <arm>_streams.json   (needs a live server)
  diverge       compare bf16 vs int4 streams -> first-divergence -> divergence.json
  wandb         push recoverable_set.json + divergence.json to W&B group
                int4ar-aime-recoverable-ubel
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

from research.downstream_quality_aime.aime_eval import (  # noqa: E402
    AIME_INSTRUCTION,
    extract_answer,
    load_aime,
)

# --------------------------------------------------------------------------- #
# 1. parse #650 .out -> per-problem correctness
# --------------------------------------------------------------------------- #
_LINE_RE = re.compile(r"id=(?P<id>\S+)\s+gold=(?P<gold>\S+)\s+maj=(?P<maj>\S+)\s+\((?P<flag>OK|x)\)")


def parse_out(path: Path) -> dict[str, dict[str, Any]]:
    """id -> {gold, maj, correct} from an aime_eval .out log."""
    out: dict[str, dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        m = _LINE_RE.search(line)
        if not m:
            continue
        out[m["id"]] = {
            "gold": m["gold"],
            "maj": m["maj"],
            "correct": m["flag"] == "OK",
        }
    return out


# --------------------------------------------------------------------------- #
# 2. concentration tagging
# --------------------------------------------------------------------------- #
def parse_number(pid: str) -> int | None:
    """Trailing integer of an AIME id is the problem number (1..15)."""
    m = re.search(r"(\d+)$", pid)
    return int(m.group(1)) if m else None


def difficulty_tier(num: int | None) -> str:
    if num is None:
        return "unknown"
    if num <= 5:
        return "easy(1-5)"
    if num <= 10:
        return "medium(6-10)"
    return "hard(11-15)"


# Coarse, *cheaply derivable* type from keyword hits on the problem text. AIME
# problems mix topics; this is a screening proxy, not ground truth — the
# problem-number tier is the more reliable difficulty axis.
_TYPE_KEYWORDS: dict[str, list[str]] = {
    "geometry": [
        "triangle", "circle", "square", "rectangle", "polygon", "angle", "area",
        "perimeter", "vertex", "vertices", "radius", "diameter", "parallel",
        "perpendicular", "sphere", "cube", "coordinate", "segment", "tangent",
        "hexagon", "rhombus", "trapezoid", "cylinder", "cone", "circumcircle",
        "incircle", "altitude", "midpoint", "quadrilateral", "isosceles",
    ],
    "number_theory": [
        "divisor", "prime", "modulo", "remainder", "divisible", "gcd", "lcm",
        "digit", "base-", "factor", "multiple of", "congruent", "integer solutions",
        "relatively prime", "positive integers", "least", "greatest",
    ],
    "combinatorics": [
        "number of ways", "choose", "permutation", "combination", "probability",
        "arrange", "subset", "select", "ordered", "distinct ways", "how many",
        "expected", "rolls", "coin", "deck", "grid",
    ],
    "algebra": [
        "polynomial", "equation", "root", "real number", "complex number",
        "sequence", "geometric", "arithmetic", "logarithm", "function", "system of",
        "sum of", "product of", "value of", "x and y",
    ],
}


def classify_type(text: str) -> tuple[str, dict[str, int]]:
    t = text.lower()
    scores = {k: sum(t.count(w) for w in kws) for k, kws in _TYPE_KEYWORDS.items()}
    best = max(scores, key=lambda k: scores[k])
    if scores[best] == 0:
        return "unclassified", scores
    return best, scores


def cmd_recoverable(args: argparse.Namespace) -> int:
    int4 = parse_out(args.int4_out)
    bf16 = parse_out(args.bf16_out)
    ids = sorted(set(int4) & set(bf16))
    print(f"[rec] int4 ids={len(int4)} bf16 ids={len(bf16)} common={len(ids)}", flush=True)

    n = len(ids)
    int4_acc = sum(int4[i]["correct"] for i in ids) / n
    bf16_acc = sum(bf16[i]["correct"] for i in ids) / n
    print(f"[rec] int4 acc={int4_acc:.4f} ({sum(int4[i]['correct'] for i in ids)}/{n})  "
          f"bf16 acc={bf16_acc:.4f} ({sum(bf16[i]['correct'] for i in ids)}/{n})", flush=True)

    R = [i for i in ids if bf16[i]["correct"] and not int4[i]["correct"]]
    rev = [i for i in ids if int4[i]["correct"] and not bf16[i]["correct"]]
    both = [i for i in ids if int4[i]["correct"] and bf16[i]["correct"]]
    neither = [i for i in ids if not int4[i]["correct"] and not bf16[i]["correct"]]
    print(f"[rec] |R| (bf16 right, int4 wrong) = {len(R)}", flush=True)
    print(f"[rec] |reverse| (int4 right, bf16 wrong) = {len(rev)}  -> {rev}", flush=True)
    print(f"[rec] both-right={len(both)} neither={len(neither)}", flush=True)
    # net deficit = |R| - |reverse| = bf16_correct - int4_correct
    net = len(R) - len(rev)
    print(f"[rec] net deficit |R|-|reverse| = {net}  (= bf16 {sum(bf16[i]['correct'] for i in ids)} - "
          f"int4 {sum(int4[i]['correct'] for i in ids)})", flush=True)
    # flips needed: int4 0.400 -> bar 0.420 needs +0.020*60 = 1.2 -> 2 problems;
    # -> bf16 0.4833 needs +0.083*60 = 5 problems.
    print("[rec] flips of R needed: to bar 0.420 -> ceil(1.2)=2 ; to bf16 0.4833 -> 5", flush=True)

    # metadata + concentration
    problems = {}
    if args.tag_type:
        plist = load_aime([y.strip() for y in args.years.split(",") if y.strip()])
        problems = {p["id"]: p for p in plist}
        print(f"[rec] loaded {len(problems)} problem texts for type tagging", flush=True)

    def tag(pid: str) -> dict[str, Any]:
        num = parse_number(pid)
        year = bf16.get(pid, {}).get("year") or _year_of(pid)
        rec: dict[str, Any] = {
            "id": pid,
            "year": year,
            "number": num,
            "tier": difficulty_tier(num),
            "gold": bf16[pid]["gold"],
            "int4_maj": int4[pid]["maj"],
            "bf16_maj": bf16[pid]["maj"],
        }
        if pid in problems:
            ptype, scores = classify_type(problems[pid]["problem"])
            rec["type"] = ptype
            rec["type_scores"] = scores
        return rec

    R_meta = [tag(i) for i in R]
    rev_meta = [tag(i) for i in rev]

    # concentration summaries over R
    def dist(key: str, items: list[dict[str, Any]]) -> dict[str, int]:
        d: dict[str, int] = {}
        for it in items:
            d[str(it.get(key))] = d.get(str(it.get(key)), 0) + 1
        return dict(sorted(d.items()))

    conc = {
        "by_year": dist("year", R_meta),
        "by_tier": dist("tier", R_meta),
        "by_number": dist("number", R_meta),
    }
    if args.tag_type:
        conc["by_type"] = dist("type", R_meta)
    print(f"[rec] R concentration: {json.dumps(conc)}", flush=True)

    out = {
        "created_at": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
        "budget": 12288,
        "n_problems": n,
        "int4_acc": int4_acc,
        "bf16_acc": bf16_acc,
        "int4_n_correct": sum(int4[i]["correct"] for i in ids),
        "bf16_n_correct": sum(bf16[i]["correct"] for i in ids),
        "R": R,
        "reverse": rev,
        "both_right": both,
        "neither": neither,
        "net_deficit": net,
        "flips_to_bar": 2,
        "flips_to_bf16": 5,
        "R_meta": R_meta,
        "reverse_meta": rev_meta,
        "concentration": conc,
        "per_problem": {i: {"int4": int4[i], "bf16": bf16[i]} for i in ids},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"[rec] wrote {args.out}", flush=True)
    return 0


def _year_of(pid: str) -> str:
    if pid.startswith("2024"):
        return "2024"
    if pid.startswith("2025-I-"):
        return "2025-I"
    if pid.startswith("2025-II-"):
        return "2025-II"
    return "unknown"


# --------------------------------------------------------------------------- #
# 3. harvest greedy token streams (per served arm)
# --------------------------------------------------------------------------- #
def greedy_logprobs(base_url: str, model: str, problem: str, *, max_tokens: int,
                    seed: int, timeout_s: int, top_logprobs: int = 5,
                    enable_thinking: bool = False, min_tokens: int = 8) -> dict[str, Any]:
    # Match the #650 greedy mt12288 request EXACTLY: --no-thinking, min_tokens=8,
    # k=1, temperature=0 (pure argmax). enable_thinking=True flips the trajectory
    # (it injects the thinking channel) and was the cause of the first-smoke
    # 94-vs-104 mismatch. logprobs only reads the already-computed logits under
    # VLLM_BATCH_INVARIANT=1, so it does not move the argmax.
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": f"{problem}\n\n{AIME_INSTRUCTION}"}],
        "n": 1,
        "temperature": 0,
        "top_p": 1.0,
        "max_tokens": max_tokens,
        "min_tokens": min_tokens,
        "seed": seed,
        "stream": False,
        "logprobs": True,
        "top_logprobs": top_logprobs,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        return json.loads(r.read().decode())


def _compact_tokens(lp_content: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per position keep chosen token (t,b,lp) + runner-up logprob r (margin)."""
    toks: list[dict[str, Any]] = []
    for e in lp_content:
        chosen_t = e["token"]
        runner = None
        for alt in e.get("top_logprobs", []):
            if alt["token"] != chosen_t:
                runner = alt["logprob"]
                break
        toks.append({"t": chosen_t, "b": e.get("bytes"), "lp": e["logprob"], "r": runner})
    return toks


def cmd_harvest(args: argparse.Namespace) -> int:
    rec = json.loads(args.recoverable.read_text())
    target_ids = list(rec["R"])
    if args.include_reverse:
        target_ids += list(rec["reverse"])
    if args.limit is not None:
        target_ids = target_ids[: args.limit]
    plist = load_aime([y.strip() for y in args.years.split(",") if y.strip()])
    pmap = {p["id"]: p for p in plist}

    valid_ids = [pid for pid in target_ids if pid in pmap]
    for pid in target_ids:
        if pid not in pmap:
            print(f"[harvest] WARN id {pid} not in dataset; skip", flush=True)

    def _one(pid: str) -> tuple[str, dict[str, Any]]:
        prob = pmap[pid]
        t0 = time.time()
        resp = greedy_logprobs(args.base_url, args.model, prob["problem"],
                               max_tokens=args.max_tokens, seed=args.seed,
                               timeout_s=args.request_timeout_s,
                               enable_thinking=args.enable_thinking,
                               min_tokens=args.min_tokens)
        ch = resp["choices"][0]
        text = ch.get("message", {}).get("content") or ""
        toks = _compact_tokens(ch.get("logprobs", {}).get("content") or [])
        ans = extract_answer(text)
        gold = prob["answer"]
        correct = ans is not None and ans == gold
        exp = rec["per_problem"][pid][args.arm]["correct"]
        repro = correct == exp
        rec_pid = {
            "id": pid, "gold": gold, "answer": ans, "correct": correct,
            "expected_correct_650": exp, "reproduced_650": repro,
            "finish_reason": ch.get("finish_reason"), "n_tokens": len(toks),
            "text_chars": len(text), "tokens": toks,
            "text": text if args.save_text else None,
        }
        print(f"[harvest:{args.arm}] id={pid} ans={ans} gold={gold} correct={correct} "
              f"exp={exp} repro={repro} ntok={len(toks)} finish={ch.get('finish_reason')} "
              f"{time.time()-t0:.1f}s", flush=True)
        return pid, rec_pid

    # Batch-invariant stack (VLLM_BATCH_INVARIANT=1): a request's greedy tokens do
    # not depend on what else is in its batch, so client concurrency yields the
    # same streams as sequential. #650 ran these arms at concurrency 16 too.
    streams: dict[str, Any] = {}
    if args.client_concurrency <= 1:
        for pid in valid_ids:
            k, v = _one(pid)
            streams[k] = v
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=args.client_concurrency) as ex:
            futs = {ex.submit(_one, pid): pid for pid in valid_ids}
            for fut in as_completed(futs):
                k, v = fut.result()
                streams[k] = v
    streams = {pid: streams[pid] for pid in valid_ids if pid in streams}
    n_reproduced = sum(int(v["reproduced_650"]) for v in streams.values())

    out = {
        "arm": args.arm,
        "base_url": args.base_url,
        "model": args.model,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "created_at": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
        "n": len(streams),
        "n_reproduced_650": n_reproduced,
        "streams": streams,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"[harvest:{args.arm}] wrote {args.out}  reproduced {n_reproduced}/{len(streams)} vs #650", flush=True)
    return 0


# --------------------------------------------------------------------------- #
# 4. first-divergence
# --------------------------------------------------------------------------- #
def _tok_key(tok: dict[str, Any]) -> tuple:
    return (tok["t"], tuple(tok["b"]) if tok.get("b") is not None else None)


def first_divergence(bf16_toks: list[dict[str, Any]], int4_toks: list[dict[str, Any]]) -> int | None:
    n = min(len(bf16_toks), len(int4_toks))
    for i in range(n):
        if _tok_key(bf16_toks[i]) != _tok_key(int4_toks[i]):
            return i
    if len(bf16_toks) != len(int4_toks):
        return n  # identical prefix, one is a strict prefix of the other
    return None  # identical streams


def locus_bucket(frac: float) -> str:
    if frac < 0.25:
        return "EARLY"
    if frac > 0.75:
        return "LATE"
    return "MID"


def cmd_diverge(args: argparse.Namespace) -> int:
    rec = json.loads(args.recoverable.read_text())
    bf16 = json.loads(args.bf16.read_text())["streams"]
    int4 = json.loads(args.int4.read_text())["streams"]

    rows: list[dict[str, Any]] = []
    for pid in rec["R"]:
        if pid not in bf16 or pid not in int4:
            print(f"[div] WARN missing streams for {pid}; skip", flush=True)
            continue
        bt = bf16[pid]["tokens"]
        it = int4[pid]["tokens"]
        d = first_divergence(bt, it)
        lb, li = len(bt), len(it)
        row: dict[str, Any] = {
            "id": pid,
            "len_bf16": lb,
            "len_int4": li,
            "div_idx": d,
            "bf16_correct": bf16[pid]["correct"],
            "int4_correct": int4[pid]["correct"],
            "bf16_repro": bf16[pid]["reproduced_650"],
            "int4_repro": int4[pid]["reproduced_650"],
        }
        if d is None:
            row["note"] = "identical_streams_unexpected"
            row["frac_bf16"] = None
            row["bucket"] = None
        else:
            row["frac_bf16"] = d / lb if lb else None
            row["frac_int4"] = d / li if li else None
            row["frac_mean"] = d / ((lb + li) / 2) if (lb + li) else None
            row["bucket"] = locus_bucket(row["frac_bf16"]) if row["frac_bf16"] is not None else None
            # decisiveness at divergence: bf16 margin (chosen - runnerup) in nats
            bd = bt[d] if d < lb else None
            idd = it[d] if d < li else None
            row["bf16_div_token"] = bd["t"] if bd else None
            row["int4_div_token"] = idd["t"] if idd else None
            row["bf16_div_margin"] = (bd["lp"] - bd["r"]) if bd and bd.get("r") is not None else None
            row["int4_div_margin"] = (idd["lp"] - idd["r"]) if idd and idd.get("r") is not None else None
        rows.append(row)
        print(f"[div] id={pid} div_idx={d} frac_bf16={row.get('frac_bf16')} "
              f"bucket={row.get('bucket')} len_bf16={lb} len_int4={li} "
              f"bf16tok={row.get('bf16_div_token')!r} int4tok={row.get('int4_div_token')!r}", flush=True)

    valid = [r for r in rows if r.get("bucket")]
    buckets = {"EARLY": 0, "MID": 0, "LATE": 0}
    for r in valid:
        buckets[r["bucket"]] += 1
    fracs = [r["frac_bf16"] for r in valid if r["frac_bf16"] is not None]
    summary = {
        "n_R": len(rec["R"]),
        "n_analyzed": len(rows),
        "n_valid": len(valid),
        "buckets": buckets,
        "frac_bf16_values": fracs,
        "frac_bf16_mean": (sum(fracs) / len(fracs)) if fracs else None,
        "frac_bf16_median": (sorted(fracs)[len(fracs) // 2]) if fracs else None,
        "n_bf16_reproduced": sum(int(bf16[p]["reproduced_650"]) for p in rec["R"] if p in bf16),
        "n_int4_reproduced": sum(int(int4[p]["reproduced_650"]) for p in rec["R"] if p in int4),
    }
    print(f"[div] SUMMARY {json.dumps(summary)}", flush=True)

    out = {"created_at": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
           "rows": rows, "summary": summary}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"[div] wrote {args.out}", flush=True)
    return 0


# --------------------------------------------------------------------------- #
# 5. verdict + W&B
# --------------------------------------------------------------------------- #
def compute_verdict(rec: dict[str, Any], div: dict[str, Any]) -> dict[str, Any]:
    """Transparent rule mapping (concentration, divergence locus) -> verdict.

    PR #668 definitions:
      RECOVERABLE_CONCENTRATED — few clustered R, predominantly LATE divergence.
      RECOVERABLE_DIFFUSE      — R spread, mixed divergence.
      FUNDAMENTAL              — early divergence on most of R, broad.
    """
    s = div["summary"]
    nv = s["n_valid"] or 1
    b = s["buckets"]
    early_frac = b["EARLY"] / nv
    late_frac = b["LATE"] / nv
    mid_frac = b["MID"] / nv

    # spread: R distributed across years and difficulty tiers without a single
    # bucket dominating (>60%). R is "clustered" only if one year or tier holds
    # the majority.
    conc = rec["concentration"]
    def _dominated(d: dict[str, int]) -> bool:
        tot = sum(d.values()) or 1
        return max(d.values()) / tot > 0.60
    year_clustered = _dominated(conc["by_year"])
    tier_clustered = _dominated(conc["by_tier"])
    clustered = year_clustered or tier_clustered

    if early_frac >= 0.50:
        verdict = "FUNDAMENTAL"
    elif late_frac >= 0.50 and clustered:
        verdict = "RECOVERABLE_CONCENTRATED"
    elif late_frac >= 0.50 and not clustered:
        # predominantly late but spread: recoverable but a targeted bump must span
        # the spread -> diffuse-leaning recoverable.
        verdict = "RECOVERABLE_DIFFUSE"
    else:
        verdict = "RECOVERABLE_DIFFUSE" if late_frac >= early_frac else "FUNDAMENTAL"
    return {
        "verdict": verdict,
        "early_frac": early_frac, "mid_frac": mid_frac, "late_frac": late_frac,
        "year_clustered": year_clustered, "tier_clustered": tier_clustered,
        "clustered": clustered,
    }


def finish_length_rate(streams: dict[str, Any], ids: list[str]) -> float:
    present = [streams[i] for i in ids if i in streams]
    if not present:
        return 0.0
    return sum(1 for s in present if s["finish_reason"] == "length") / len(present)


# --------------------------------------------------------------------------- #
# 5b. cross-condition reproducibility / session-noise of the int4 deficit
# --------------------------------------------------------------------------- #
def cmd_reproducibility(args: argparse.Namespace) -> int:
    """How stable is the int4-AR greedy AIME result across serving conditions?

    The recoverable set R is defined by a SINGLE #650 int4 run. This quantifies
    how much of the apparent int4 deficit is near-tie serving noise vs a stable
    quantization gap, by comparing the #650 transcript against fresh re-serves:
      * boundary (R+reverse) per-problem ANSWER stability across three conditions
        (#650 conc16/old-proc, my conc11, my conc1=batch-of-1);
      * batch-flip rate = boundary problems whose argmax answer changes between
        conc1 and conc11 in the SAME process (isolates batch/chunk sensitivity);
      * if a fresh full-60 int4 run is supplied: aggregate score stability and
        R-set membership reproduction vs #650.
    """
    rec = json.loads(args.recoverable.read_text())
    R = list(rec["R"]); rev = list(rec["reverse"]); boundary = R + rev
    pp = rec["per_problem"]
    c11 = json.loads(args.c11.read_text())["streams"] if args.c11 and args.c11.exists() else {}
    c1 = json.loads(args.c1.read_text())["streams"] if args.c1 and args.c1.exists() else {}

    def ans(d: dict[str, Any], pid: str) -> str:
        return str(d.get(pid, {}).get("answer"))

    boundary_rows = []
    all3 = c1c11 = c1_650 = c11_650 = batch_flip = 0
    for pid in boundary:
        a650 = str(pp[pid]["int4"]["maj"]); a11 = ans(c11, pid); a1 = ans(c1, pid)
        o650 = bool(pp[pid]["int4"]["correct"])
        o11 = bool(c11.get(pid, {}).get("correct")); o1 = bool(c1.get(pid, {}).get("correct"))
        eq3 = a650 == a11 == a1
        all3 += eq3; c1c11 += (a1 == a11); c1_650 += (a1 == a650); c11_650 += (a11 == a650)
        batch_flip += (a1 != a11)
        boundary_rows.append({"id": pid, "set": "R" if pid in R else "rev",
                              "gold": pp[pid]["bf16"]["gold"],
                              "ans_650": a650, "ans_c11": a11, "ans_c1": a1,
                              "ok_650": o650, "ok_c11": o11, "ok_c1": o1})
    nb = len(boundary)
    repro = {
        "created_at": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
        "n_boundary": nb,
        "boundary_answer_all3_equal": all3,
        "boundary_answer_c1_eq_c11": c1c11,
        "boundary_answer_c1_eq_650": c1_650,
        "boundary_answer_c11_eq_650": c11_650,
        "boundary_batch_flip_count": batch_flip,
        "boundary_batch_flip_rate": batch_flip / nb if nb else None,
        "boundary_correct_650": sum(r["ok_650"] for r in boundary_rows),
        "boundary_correct_c11": sum(r["ok_c11"] for r in boundary_rows),
        "boundary_correct_c1": sum(r["ok_c1"] for r in boundary_rows),
        "boundary_rows": boundary_rows,
    }

    # full-60 session noise (optional): parse a fresh int4 aime_eval .out + the
    # #650 int4/bf16 .out files, recompute aggregate + R membership stability.
    if args.fresh60_out and args.fresh60_out.exists():
        fresh = parse_out(args.fresh60_out)
        int4_650 = parse_out(args.int4_650_out)
        bf16_650 = parse_out(args.bf16_650_out)
        ids60 = sorted(set(fresh) & set(int4_650) & set(bf16_650))
        n60 = len(ids60)
        fresh_nc = sum(fresh[i]["correct"] for i in ids60)
        i650_nc = sum(int4_650[i]["correct"] for i in ids60)
        b650_nc = sum(bf16_650[i]["correct"] for i in ids60)
        # per-problem cross-session correctness flips (fresh vs #650 int4)
        flips = [i for i in ids60 if fresh[i]["correct"] != int4_650[i]["correct"]]
        ans_flips = [i for i in ids60 if fresh[i]["maj"] != int4_650[i]["maj"]]
        # R recomputed with the fresh int4 run as the int4 arm
        R_fresh = [i for i in ids60 if bf16_650[i]["correct"] and not fresh[i]["correct"]]
        R_set, Rf_set = set(R), set(R_fresh)
        repro["fresh60"] = {
            "n": n60,
            "int4_fresh_n_correct": fresh_nc, "int4_fresh_acc": fresh_nc / n60 if n60 else None,
            "int4_650_n_correct": i650_nc, "int4_650_acc": i650_nc / n60 if n60 else None,
            "bf16_650_n_correct": b650_nc, "bf16_650_acc": b650_nc / n60 if n60 else None,
            "crosssession_correctness_flips": len(flips),
            "crosssession_answer_flips": len(ans_flips),
            "flip_ids": flips,
            "R_650_size": len(R_set), "R_fresh_size": len(Rf_set),
            "R_overlap": len(R_set & Rf_set),
            "R_jaccard": len(R_set & Rf_set) / len(R_set | Rf_set) if (R_set | Rf_set) else None,
            "net_deficit_650": b650_nc - i650_nc,
            "net_deficit_fresh": b650_nc - fresh_nc,
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(repro, indent=2))
    print(f"[repro] boundary n={nb}: all3_equal={all3} c1==c11={c1c11} "
          f"batch_flip={batch_flip}/{nb}  correct 650/c11/c1="
          f"{repro['boundary_correct_650']}/{repro['boundary_correct_c11']}/{repro['boundary_correct_c1']}",
          flush=True)
    if "fresh60" in repro:
        f = repro["fresh60"]
        print(f"[repro] fresh60: int4 {f['int4_fresh_n_correct']}/{f['n']} (acc {f['int4_fresh_acc']:.4f}) "
              f"vs #650 {f['int4_650_n_correct']}/{f['n']}; xsession correctness flips={f['crosssession_correctness_flips']} "
              f"answer flips={f['crosssession_answer_flips']}; R overlap {f['R_overlap']}/{f['R_650_size']} "
              f"(jaccard {f['R_jaccard']:.3f})", flush=True)
    print(f"[repro] wrote {args.out}", flush=True)
    return 0


def cmd_wandb(args: argparse.Namespace) -> int:
    import os
    import wandb

    rec = json.loads(args.recoverable.read_text())
    div = json.loads(args.divergence.read_text())
    bf16 = json.loads(args.bf16.read_text())["streams"] if args.bf16 else {}
    int4 = json.loads(args.int4.read_text())["streams"] if args.int4 else {}
    repro = (json.loads(args.reproducibility.read_text())
             if args.reproducibility and args.reproducibility.exists() else {})

    verdict = compute_verdict(rec, div)
    R = rec["R"]
    flr_int4 = finish_length_rate(int4, R)
    flr_bf16 = finish_length_rate(bf16, R)

    entity = os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team")
    project = os.environ.get("WANDB_PROJECT", "gemma-challenge-senpai")
    group = "int4ar-aime-recoverable-ubel"

    def _rid(name: str) -> str:
        s = re.sub(r"[^A-Za-z0-9_-]", "-", name.replace("ubel/", ""))
        return f"pr668-{s}"

    common = {
        "analysis_only": True, "official_tps": 0, "pr": 668, "student": "ubel",
        "budget": rec["budget"], "engine": "vllm==0.22.0",
        "model_int4": "int4_g128_lmhead", "model_bf16": "gemma-4-E4B-it",
        "no_thinking": True, "min_tokens": 8, "seed": 0,
    }
    ids: list[str] = []

    # --- run 1: recoverable set + concentration -----------------------------
    nm = "ubel/recoverable-aime-set"
    run = wandb.init(project=project, entity=entity, group=group, reinit=True,
                     id=_rid(nm), resume="allow", name=nm, job_type="recoverable-set",
                     config={**common})
    set_scalars = {
        "n_problems": rec["n_problems"], "int4_acc": rec["int4_acc"], "bf16_acc": rec["bf16_acc"],
        "int4_n_correct": rec["int4_n_correct"], "bf16_n_correct": rec["bf16_n_correct"],
        "R_size": len(rec["R"]), "reverse_size": len(rec["reverse"]),
        "net_deficit": rec["net_deficit"], "flips_to_bar": rec["flips_to_bar"],
        "flips_to_bf16": rec["flips_to_bf16"], "bar": 0.420,
    }
    for c in ("by_year", "by_tier", "by_type"):
        for k, v in rec["concentration"].get(c, {}).items():
            set_scalars[f"{c}.{k}"] = v
    for k, v in set_scalars.items():
        run.summary[k] = v
        if isinstance(v, (int, float, bool)):
            wandb.log({k: v})
    cols = ["id", "year", "number", "tier", "type", "gold", "int4_maj", "bf16_maj"]
    tbl = wandb.Table(columns=cols)
    for m in rec["R_meta"]:
        tbl.add_data(*[m.get(c) for c in cols])
    run.log({"recoverable_set_R": tbl})
    run.summary["R_ids"] = ",".join(rec["R"])
    run.summary["reverse_ids"] = ",".join(rec["reverse"])
    ids.append(run.id); run.finish()

    # --- run 2: first-divergence locus --------------------------------------
    nm = "ubel/recoverable-aime-divergence"
    run = wandb.init(project=project, entity=entity, group=group, reinit=True,
                     id=_rid(nm), resume="allow", name=nm, job_type="divergence",
                     config={**common})
    s = div["summary"]
    div_scalars = {
        "n_R": s["n_R"], "n_analyzed": s["n_analyzed"], "n_valid": s["n_valid"],
        "early": s["buckets"]["EARLY"], "mid": s["buckets"]["MID"], "late": s["buckets"]["LATE"],
        "early_frac": verdict["early_frac"], "mid_frac": verdict["mid_frac"],
        "late_frac": verdict["late_frac"],
        "frac_bf16_mean": s["frac_bf16_mean"], "frac_bf16_median": s["frac_bf16_median"],
        "n_bf16_reproduced": s["n_bf16_reproduced"], "n_int4_reproduced": s["n_int4_reproduced"],
        "finish_length_rate_int4": flr_int4, "finish_length_rate_bf16": flr_bf16,
        "finish_length_rate": max(flr_int4, flr_bf16),
    }
    for k, v in div_scalars.items():
        if v is not None:
            run.summary[k] = v
            if isinstance(v, (int, float, bool)):
                wandb.log({k: v})
    dcols = ["id", "div_idx", "frac_bf16", "frac_int4", "bucket", "len_bf16", "len_int4",
             "bf16_div_token", "int4_div_token", "bf16_div_margin", "int4_div_margin",
             "bf16_repro", "int4_repro"]
    dtbl = wandb.Table(columns=dcols)
    for r in div["rows"]:
        dtbl.add_data(*[r.get(c) for c in dcols])
    run.log({"divergence_rows": dtbl})
    ids.append(run.id); run.finish()

    # --- run 3: VERDICT (machine-checkable guards live here) ------------------
    nm = "ubel/recoverable-aime-VERDICT"
    run = wandb.init(project=project, entity=entity, group=group, reinit=True,
                     id=_rid(nm), resume="allow", name=nm, job_type="verdict",
                     config={**common})
    vscalars = {
        "analysis_only": True, "official_tps": 0,
        "R_size": len(rec["R"]), "net_deficit": rec["net_deficit"],
        "flips_to_bar": rec["flips_to_bar"], "flips_to_bf16": rec["flips_to_bf16"],
        "early": s["buckets"]["EARLY"], "mid": s["buckets"]["MID"], "late": s["buckets"]["LATE"],
        "early_frac": verdict["early_frac"], "late_frac": verdict["late_frac"],
        "clustered": verdict["clustered"],
        "finish_length_rate": max(flr_int4, flr_bf16),
        "n_bf16_reproduced": s["n_bf16_reproduced"], "n_int4_reproduced": s["n_int4_reproduced"],
    }
    # session-noise / reproducibility guards (the int4 deficit is near-tie noise)
    if repro:
        vscalars.update({
            "repro_boundary_n": repro["n_boundary"],
            "repro_boundary_answer_all3_equal": repro["boundary_answer_all3_equal"],
            "repro_boundary_answer_c1_eq_c11": repro["boundary_answer_c1_eq_c11"],
            "repro_boundary_batch_flip_count": repro["boundary_batch_flip_count"],
            "repro_boundary_batch_flip_rate": repro["boundary_batch_flip_rate"],
        })
        f = repro.get("fresh60")
        if f:
            vscalars.update({
                "repro_int4_fresh_n_correct": f["int4_fresh_n_correct"],
                "repro_int4_fresh_acc": f["int4_fresh_acc"],
                "repro_int4_650_n_correct": f["int4_650_n_correct"],
                "repro_bf16_650_n_correct": f["bf16_650_n_correct"],
                "repro_xsession_correctness_flips": f["crosssession_correctness_flips"],
                "repro_xsession_answer_flips": f["crosssession_answer_flips"],
                "repro_R_overlap": f["R_overlap"], "repro_R_650_size": f["R_650_size"],
                "repro_R_fresh_size": f["R_fresh_size"], "repro_R_jaccard": f["R_jaccard"],
                "repro_net_deficit_650": f["net_deficit_650"],
                "repro_net_deficit_fresh": f["net_deficit_fresh"],
            })
    for k, v in vscalars.items():
        run.summary[k] = v
        if isinstance(v, (int, float, bool)):
            wandb.log({k: v})
    run.summary["verdict"] = verdict["verdict"]
    ids.append(run.id); run.finish()

    print(f"[wandb] logged {len(ids)} runs -> group {group}  ids={ids}", flush=True)
    print(f"[wandb] VERDICT = {verdict['verdict']}  "
          f"(early={verdict['early_frac']:.2f} late={verdict['late_frac']:.2f} "
          f"clustered={verdict['clustered']})", flush=True)
    out = {"group": group, "wandb_run_ids": ids, "verdict": verdict,
           "finish_length_rate_int4": flr_int4, "finish_length_rate_bf16": flr_bf16}
    (HERE / "wandb_runs.json").write_text(json.dumps(out, indent=2))
    return 0


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("recoverable", help="parse #650 .out -> R + concentration")
    p.add_argument("--int4-out", type=Path, required=True)
    p.add_argument("--bf16-out", type=Path, required=True)
    p.add_argument("--years", default="2024,2025-I,2025-II")
    p.add_argument("--tag-type", action="store_true", help="load problem texts and tag coarse type (network)")
    p.add_argument("--out", type=Path, default=HERE / "recoverable_set.json")
    p.set_defaults(func=cmd_recoverable)

    p = sub.add_parser("harvest", help="greedy+logprobs token streams from a served arm")
    p.add_argument("--arm", choices=["int4", "bf16"], required=True)
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--model", default="gemma-4-e4b-it")
    p.add_argument("--recoverable", type=Path, default=HERE / "recoverable_set.json")
    p.add_argument("--years", default="2024,2025-I,2025-II")
    p.add_argument("--max-tokens", type=int, default=12288)
    p.add_argument("--min-tokens", type=int, default=8, help="match #650 (min_tokens=8)")
    p.add_argument("--enable-thinking", action="store_true",
                   help="#650 greedy used --no-thinking; leave OFF to reproduce it")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--client-concurrency", type=int, default=8,
                   help="parallel requests (safe on batch-invariant stack; #650 used 16)")
    p.add_argument("--include-reverse", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="cap number of problems (smoke)")
    p.add_argument("--save-text", action="store_true")
    p.add_argument("--request-timeout-s", type=int, default=1800)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_harvest)

    p = sub.add_parser("diverge", help="first-divergence from bf16+int4 streams")
    p.add_argument("--bf16", type=Path, required=True)
    p.add_argument("--int4", type=Path, required=True)
    p.add_argument("--recoverable", type=Path, default=HERE / "recoverable_set.json")
    p.add_argument("--out", type=Path, default=HERE / "divergence.json")
    p.set_defaults(func=cmd_diverge)

    p = sub.add_parser("reproducibility", help="cross-condition / session-noise of the int4 deficit")
    p.add_argument("--recoverable", type=Path, default=HERE / "recoverable_set.json")
    p.add_argument("--c11", type=Path, default=HERE / "int4_streams.json", help="int4 conc-11 streams")
    p.add_argument("--c1", type=Path, default=HERE / "int4_streams_seq.json", help="int4 conc-1 streams")
    p.add_argument("--fresh60-out", type=Path, default=HERE / "_fresh60_int4.out",
                   help="fresh full-60 int4 aime_eval stdout (optional)")
    p.add_argument("--int4-650-out", type=Path,
                   default=REPO / "research/validity/int4ar_denom_harden/_aime_int4ar_mt12288.out")
    p.add_argument("--bf16-650-out", type=Path,
                   default=REPO / "research/validity/int4ar_denom_harden/_aime_bf16_mt12288.out")
    p.add_argument("--out", type=Path, default=HERE / "reproducibility.json")
    p.set_defaults(func=cmd_reproducibility)

    p = sub.add_parser("wandb", help="log recoverable_set + divergence + verdict to W&B")
    p.add_argument("--recoverable", type=Path, default=HERE / "recoverable_set.json")
    p.add_argument("--divergence", type=Path, default=HERE / "divergence.json")
    p.add_argument("--bf16", type=Path, default=HERE / "bf16_streams.json")
    p.add_argument("--int4", type=Path, default=HERE / "int4_streams.json")
    p.add_argument("--reproducibility", type=Path, default=HERE / "reproducibility.json")
    p.set_defaults(func=cmd_wandb)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
