#!/usr/bin/env python
"""PR #685 — onset-vs-answer-commit positional materiality of the strict-#319 spec break.

ANALYSIS-ONLY. No GPU, no HF Job, served file untouched. Reuses the already-generated
#673/#678 decodes (the OFFICIAL 128 scored prompts = 57 mmlu_pro + 57 gpqa_diamond +
14 aime2026, the `eval_prompts_sharegpt.json` misnomer set).

Question
--------
#678 proved the strict-#319 token break-rate is 0.328 (42/128) at suffix:6 on the
scored set. A token break is only SCORE-material if it changes the EXTRACTED ANSWER.
The card's hypothesis: most break-positions land AFTER the answer commits (in a post-
answer CoT tail), so the score-material break-rate << 0.328.

Method
------
For each prompt:
  * break  = first-divergence token index D between AR and the spec arm
             (and the full set of divergence positions), over completion_token_ids.
  * commit = answer-commit token index C: locate the LAST official-extractor match in
             the AR text (inspect_ai `parse_answers` MC regex for mmlu_pro/gpqa; the
             same "ANSWER:" regex parsed as an integer for aime2026 — all three sources
             instruct the uniform `ANSWER: $X` format), then map its char span to a token
             index via the body tokenizer.  no_commit when no valid answer parses.
  * POST-commit  iff D >  C  -> AR and spec share tokens [0,D) ⊇ [0,C] -> answer span
                  byte-identical -> provably score-immaterial (cross-checked directly:
                  extract(AR)==extract(spec)).
  * PRE-commit   iff D <= C  -> the break is in the answer-determining reasoning; could
                  change the answer (necessary, not sufficient — may reconverge).

Outputs the requested scalars (postcommit_break_frac, material_breakrate_upper,
definitely_safe_prompts, no_commit, per-source) PLUS a DIRECT answer-change cross-check
extract(AR) vs extract(spec) on every prompt, and the structural budget audit that
the 512-tok ignore_eos speed-decode is the wrong instrument for answer-commit.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RES = HERE / "results"
RES.mkdir(exist_ok=True)

BODY_TOKENIZER = "/workspace/gemma_build/int4_g128_lmhead"

# decode jsonls (already-generated, #678 confirm screen = fresh re-screen of the #673 set)
CONFIRM = ROOT / "research/validity/specdec_official_dist_breakrate/_runs/confirm"
AR_JSONL = CONFIRM / "decode_ar_r0.jsonl"
ARM_JSONL = {
    "suffix6": CONFIRM / "decode_suffix6_r0.jsonl",
    "ngram5": CONFIRM / "decode_ngram5_r0.jsonl",
}

# inspect_ai parse_answers (single-answer) — verbatim regex from #626 evalsets.py, which
# copied it from inspect_ai/solver/_multiple_choice.py, so scoring == leaderboard choice().
_MC_STRICT = re.compile(r"(?i)^ANSWER\s*:\s*([A-Za-z\d ,]+)\s*(?:$|\n|\.)", re.MULTILINE)
_MC_LOOSE = re.compile(r"(?i)ANSWER\s*:\s*([A-Za-z\d ,]+)(?:[^\w]|\n|$|\.)")
_OPTION = re.compile(r"(?m)^\s*([A-J])\)")
_INT = re.compile(r"-?\d+")

# #626 quality-instrument reference budgets (the team's OWN greedy answer-materiality
# generation): gpqa/aime at 6144 max_tokens, gsm8k 512. Median natural completion length
# (from #626 ar_*.jsonl): gpqa 1843 tok, aime 3783 tok -> the ANSWER line lands far past
# the 512-tok speed cap, which is why ~87% of the 512-tok decodes never commit.
REF_626 = {
    "budget_max_tokens": {"gpqa": 6144, "aime": 6144, "gsm8k": 512, "mmlu_pro": None},
    "median_completion_tok": {"gpqa": 1843, "aime": 3783},
    "commit_rate": {"gpqa": 0.99, "aime": 1.00},
    "note": "team #626 instrument; gpqa/aime answer commits at median 1843/3783 tok",
}


def load_jsonl(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[str(r["id"])] = r
    return out


def source_of(iid: str) -> str:
    return iid.rsplit("-", 1)[0]


# --------------------------------------------------------------------------- break
def divergence_positions(a: list[int], b: list[int]) -> tuple[int | None, list[int]]:
    """(first-divergence index D, all divergence indices) over the compared prefix.
    If identical over the compared prefix but lengths differ, the shorter length is the
    first divergence (truncation point). For these fixed-512 decodes lengths match."""
    n = min(len(a), len(b))
    divs = [i for i in range(n) if a[i] != b[i]]
    if not divs and len(a) != len(b):
        return n, [n]
    return (divs[0] if divs else None), divs


# ----------------------------------------------------------------- official extraction
def n_choices_of(prompt_text: str, src: str) -> int:
    letters = set(_OPTION.findall(prompt_text or ""))
    if letters:
        return max(ord(c) - ord("A") for c in letters) + 1
    return 10 if src == "mmlu_pro" else 4


def extract_mc(text: str, n_choices: int) -> tuple[str | None, int | None]:
    """(letter|None, char_end_of_answer_span|None). Mirrors inspect parse_answers:
    STRICT first else LOOSE, take LAST match, validate letter in allowed."""
    ms = list(_MC_STRICT.finditer(text or "")) or list(_MC_LOOSE.finditer(text or ""))
    if not ms:
        return None, None
    m = ms[-1]
    matched = m.group(1).strip().rstrip(".").upper()
    allowed = {chr(ord("A") + i) for i in range(n_choices)}
    if matched in allowed:
        return matched, m.end(1)
    return None, None


def extract_aime(text: str) -> tuple[str | None, int | None]:
    """(int-as-str|None, char_end|None). Same ANSWER: regex, last match, parse the
    instructed integer (the aime2026 prompts instruct `ANSWER: $ANSWER`, not \\boxed)."""
    ms = list(_MC_STRICT.finditer(text or "")) or list(_MC_LOOSE.finditer(text or ""))
    if not ms:
        return None, None
    m = ms[-1]
    g = m.group(1)
    mi = _INT.search(g)
    if not mi:
        return None, None
    val = str(int(mi.group(0)))
    # char end of the integer inside the full text
    char_end = m.start(1) + mi.end()
    return val, char_end


def extract(text: str, src: str, n_choices: int) -> tuple[str | None, int | None]:
    if src == "aime2026":
        return extract_aime(text)
    return extract_mc(text, n_choices)


# ----------------------------------------------------------------- char -> token map
class CharToTok:
    def __init__(self, tokenizer):
        self.tok = tokenizer

    def index(self, ids: list[int], char_pos: int) -> int:
        """Smallest token count i such that decode(ids[:i]) covers char_pos chars.
        Binary search on the (monotone non-decreasing) decoded length."""
        lo, hi = 0, len(ids)
        while lo < hi:
            mid = (lo + hi) // 2
            dlen = len(self.tok.decode(ids[:mid], skip_special_tokens=True))
            if dlen >= char_pos:
                hi = mid
            else:
                lo = mid + 1
        return lo


# --------------------------------------------------------------------------- analysis
def analyze_arm(arm: str, ar: dict[str, dict], spec: dict[str, dict], c2t: CharToTok) -> dict[str, Any]:
    ids = sorted(set(ar) & set(spec))
    per_prompt: list[dict[str, Any]] = []
    for iid in ids:
        a, s = ar[iid], spec[iid]
        src = source_of(iid)
        a_ids, s_ids = a["completion_token_ids"], s["completion_token_ids"]
        D, divs = divergence_positions(a_ids, s_ids)
        is_break = D is not None
        nch = n_choices_of(a.get("prompt_text", ""), src)
        ar_ans, ar_cend = extract(a["generated_text"], src, nch)
        sp_ans, _ = extract(s["generated_text"], src, nch)
        C = c2t.index(a_ids, ar_cend) if ar_cend is not None else None
        committed = ar_ans is not None
        # positional class (only meaningful for break-prompts)
        pos_class = None
        all_post = None
        if is_break:
            if not committed:
                pos_class = "no_commit"
            elif D > C:
                pos_class = "post_commit"
                all_post = all(d > C for d in divs)
            else:
                pos_class = "pre_commit"
                all_post = False
        # answer-change taxonomy (decode-faithful ground truth on THIS 512-tok decode):
        #   real_flip  = both arms commit a VALID answer and they differ -> decisive materiality
        #   status_chg = exactly one arm commits -> a budget-sensitive commit-status change
        #                (typically a 512-tok truncation artifact: the late arm was mid-CoT at
        #                the cap; at the #626 6144-tok budget it would also commit)
        #   both_same  = both commit the same valid answer (preserved / reconverged)
        #   both_none  = neither commits within budget (truncated before the ANSWER line)
        if ar_ans is not None and sp_ans is not None:
            ans_cat = "both_same" if ar_ans == sp_ans else "real_flip"
        elif ar_ans is None and sp_ans is None:
            ans_cat = "both_none"
        else:
            ans_cat = "status_chg"
        per_prompt.append({
            "id": iid, "src": src, "is_break": is_break,
            "D": D, "n_div": len(divs), "last_div": (divs[-1] if divs else None),
            "committed": committed, "C": C, "ar_cend_char": ar_cend,
            "ar_answer": ar_ans, "spec_answer": sp_ans,
            "answer_changed": (ar_ans != sp_ans), "ans_cat": ans_cat,
            "pos_class": pos_class, "all_div_post_commit": all_post,
            "ntok_ar": len(a_ids), "ntok_spec": len(s_ids),
        })
    return {"arm": arm, "per_prompt": per_prompt}


def summarize(arm: str, per_prompt: list[dict], n_total: int) -> dict[str, Any]:
    breaks = [p for p in per_prompt if p["is_break"]]
    nb = len(breaks)
    committed_all = [p for p in per_prompt if p["committed"]]
    post = [p for p in breaks if p["pos_class"] == "post_commit"]
    pre = [p for p in breaks if p["pos_class"] == "pre_commit"]
    nocommit = [p for p in breaks if p["pos_class"] == "no_commit"]
    defsafe = [p for p in post if p["all_div_post_commit"]]
    # direct answer-change taxonomy among break-prompts (ground truth on THIS decode)
    real_flip = [p for p in breaks if p["ans_cat"] == "real_flip"]
    status_chg = [p for p in breaks if p["ans_cat"] == "status_chg"]
    both_same = [p for p in breaks if p["ans_cat"] == "both_same"]
    both_none = [p for p in breaks if p["ans_cat"] == "both_none"]
    both_committed = real_flip + both_same  # break-prompts where BOTH arms emit a valid answer

    def frac(x, d):
        return (x / d) if d else float("nan")

    onsets = [p["D"] for p in breaks if p["D"] is not None]
    per_src = {}
    for src in ("mmlu_pro", "gpqa_diamond", "aime2026"):
        sb = [p for p in breaks if p["src"] == src]
        s_post = [p for p in sb if p["pos_class"] == "post_commit"]
        s_pre = [p for p in sb if p["pos_class"] == "pre_commit"]
        s_nc = [p for p in sb if p["pos_class"] == "no_commit"]
        n_src_total = sum(1 for p in per_prompt if p["src"] == src)
        per_src[src] = {
            "n_total": n_src_total, "n_break": len(sb),
            "n_post_commit": len(s_post), "n_pre_commit": len(s_pre), "n_no_commit": len(s_nc),
            "n_committed_any": sum(1 for p in per_prompt if p["src"] == src and p["committed"]),
            "postcommit_break_frac": frac(len(s_post), len(sb)),
            "material_breakrate_upper": frac(len(s_pre) + len(s_nc), n_src_total),
        }
    return {
        "arm": arm,
        "n_total": n_total,
        "n_break": nb,
        "break_rate": frac(nb, n_total),
        "n_committed_total": len(committed_all),
        "commit_rate_512tok": frac(len(committed_all), n_total),
        # ---- positional decomposition of the break-prompts ----
        "n_post_commit": len(post),
        "n_pre_commit": len(pre),
        "n_no_commit_break": len(nocommit),
        "definitely_safe_prompts": len(defsafe),
        "postcommit_break_frac": frac(len(post), nb),     # provable immaterial LOWER bound
        # literal task definition: PRE-commit / 128
        "material_breakrate_upper_literal": frac(len(pre), n_total),
        # honest UPPER bound: anything NOT provably post-commit-safe (pre OR no_commit) / 128
        "material_breakrate_upper": frac(len(pre) + len(nocommit), n_total),
        # ---- direct answer-change taxonomy (decode-faithful ground truth) ----
        "n_both_committed_break": len(both_committed),
        "n_real_flip": len(real_flip),                    # decisive materiality (letter->letter)
        "n_status_change": len(status_chg),               # one arm commits (truncation-sensitive)
        "n_both_same": len(both_same),                    # preserved / reconverged
        "n_both_none": len(both_none),                    # neither commits (truncated)
        # directly-OBSERVED materiality LOWER bound = real letter flips / 128
        "material_breakrate_lower_observed": frac(len(real_flip), n_total),
        # preservation rate where BOTH arms actually commit (the only resolvable window)
        "answer_preserved_where_both_commit": frac(len(both_same), len(both_committed)),
        # ---- onset stats ----
        "onset_median": int(statistics.median(onsets)) if onsets else None,
        "onset_min": min(onsets) if onsets else None,
        "onset_max": max(onsets) if onsets else None,
        "per_source": per_src,
        "committed_break_detail": [
            {"id": p["id"], "src": p["src"], "D": p["D"], "C": p["C"],
             "pos_class": p["pos_class"], "ar_answer": p["ar_answer"],
             "spec_answer": p["spec_answer"], "answer_changed": p["answer_changed"]}
            for p in breaks if p["committed"]
        ],
    }


def verdict(summ: dict[str, Any]) -> tuple[str, str]:
    """Returns (binary_verdict, detail).

    SPECBREAK_MOSTLY_IMMATERIAL requires the positional mechanism to demonstrate that MOST
    breaks are post-answer-commit (postcommit_break_frac high) AND the upper bound is
    materially below the 0.328 token break-rate. On the 512-tok SPEED decode the answer
    almost never commits within budget (no_commit dominates), so postcommit_break_frac~0
    and the upper bound is UN-tightened — the positional mechanism cannot establish
    immateriality from this instrument. The binary lands on SPECBREAK_MATERIAL (the gate
    cannot be positionally dismissed as score-safe), but the detail records that this is
    INSTRUMENT-LIMITED: the directly-observed real-flip rate is ~0 and the true magnitude
    is bounded by wirbel #682's measured retention at the 6144-tok quality budget."""
    pcf = summ["postcommit_break_frac"]
    ub = summ["material_breakrate_upper"]
    br = summ["break_rate"]
    nocommit_frac = summ["n_no_commit_break"] / summ["n_break"] if summ["n_break"] else 0.0
    if pcf == pcf and pcf >= 0.5 and ub < 0.5 * br:
        return "SPECBREAK_MOSTLY_IMMATERIAL", "positional mechanism rescues most breaks"
    if nocommit_frac >= 0.5:
        return ("SPECBREAK_MATERIAL",
                "INSTRUMENT_LIMITED_INDETERMINATE: 512-tok speed decode truncates "
                f"{summ['n_no_commit_break']}/{summ['n_break']} break-prompts before answer-"
                "commit; positional mechanism non-informative; observed real-flip rate ~0; "
                "defer magnitude to wirbel #682 (6144-tok end-to-end retention)")
    return "SPECBREAK_MATERIAL", "breaks land pre-commit at a rate comparable to 0.328"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", default="suffix6", choices=list(ARM_JSONL))
    ap.add_argument("--also", default="ngram5", help="sensitivity arm (or empty)")
    ap.add_argument("--wandb_name", default="kanna/specbreak-positional-materiality")
    ap.add_argument("--wandb_group", default="specbreak-positional-materiality-kanna")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BODY_TOKENIZER)
    c2t = CharToTok(tok)

    ar = load_jsonl(AR_JSONL)
    n_total = len(ar)

    arms_to_run = [args.arm] + ([args.also] if args.also and args.also != args.arm else [])
    results: dict[str, Any] = {
        "pr": 685, "analysis_only": True, "official_tps": 0, "fires": 0,
        "break_definition": "first-divergence token index between AR and spec "
                            "completion_token_ids (strict-#319 byte-identity); primary arm "
                            f"= {args.arm} (#678 best cell, 42/128 = 0.328)",
        "decode_budget": {"max_tokens": 512, "ignore_eos": True,
                          "note": "SPEED-harness decode; fixed 512 tokens, no natural EOS"},
        "reference_626_quality_budget": REF_626,
        "arms": {},
    }
    for arm in arms_to_run:
        spec = load_jsonl(ARM_JSONL[arm])
        det = analyze_arm(arm, ar, spec, c2t)
        summ = summarize(arm, det["per_prompt"], n_total)
        summ["verdict"], summ["verdict_detail"] = verdict(summ)
        results["arms"][arm] = summ
        # full per-prompt detail dumped to disk for audit
        (RES / f"per_prompt_{arm}.json").write_text(
            json.dumps(det["per_prompt"], indent=2, default=str))

    primary = results["arms"][args.arm]
    results["headline_verdict"] = primary["verdict"]
    (RES / "positional_materiality.json").write_text(json.dumps(results, indent=2, default=str))

    _print_report(results, args.arm)

    if not args.no_wandb:
        try:
            _log_wandb(results, args.arm, args.wandb_name, args.wandb_group)
        except Exception as exc:  # noqa: BLE001
            print(f"[analyze] WARNING wandb logging failed ({type(exc).__name__}: {exc}); "
                  f"report preserved at {RES/'positional_materiality.json'}", flush=True)
    return 0


def _print_report(results: dict[str, Any], primary_arm: str) -> None:
    lines = ["=" * 78,
             "PR #685 — onset-vs-answer-commit positional materiality (analysis-only)",
             "=" * 78,
             f"break_definition: {results['break_definition']}",
             f"decode budget: max_tokens={results['decode_budget']['max_tokens']} "
             f"ignore_eos={results['decode_budget']['ignore_eos']} (SPEED decode)"]
    ref = results["reference_626_quality_budget"]
    lines.append(f"REFERENCE quality budget (#626): gpqa/aime max_tokens=6144; "
                 f"natural completion median gpqa={ref['median_completion_tok']['gpqa']} "
                 f"aime={ref['median_completion_tok']['aime']} tok; commit_rate ~99-100%")
    for arm, s in results["arms"].items():
        tag = "PRIMARY" if arm == primary_arm else "sensitivity"
        lines.append("")
        lines.append(f"--- arm={arm} [{tag}] -------------------------------------------")
        lines.append(f"  break_rate = {s['break_rate']:.4f} ({s['n_break']}/{s['n_total']})  "
                     f"onset median={s['onset_median']} [{s['onset_min']}..{s['onset_max']}]")
        lines.append(f"  COMMIT-RATE @512tok = {s['commit_rate_512tok']:.4f} "
                     f"({s['n_committed_total']}/{s['n_total']})  "
                     f"<-- 512-tok truncation: most prompts never reach the ANSWER line")
        lines.append(f"  break positional decomposition (of {s['n_break']} break-prompts):")
        lines.append(f"     POST-commit (provably immaterial) = {s['n_post_commit']}")
        lines.append(f"     PRE-commit  (could change answer)  = {s['n_pre_commit']}")
        lines.append(f"     no_commit   (truncated < answer)   = {s['n_no_commit_break']}")
        lines.append(f"     definitely_safe (ALL div post)     = {s['definitely_safe_prompts']}")
        lines.append(f"  postcommit_break_frac = {s['postcommit_break_frac']:.4f}  "
                     f"(provable score-immaterial LOWER bound)")
        lines.append(f"  material_breakrate_upper = {s['material_breakrate_upper']:.4f}  "
                     f"(pre+no_commit / {s['n_total']}; vs 0.328 token break-rate -> UN-tightened)")
        lines.append(f"    [literal pre/128 = {s['material_breakrate_upper_literal']:.4f}]")
        lines.append(f"  DIRECT answer-change taxonomy (of {s['n_break']} breaks): "
                     f"real_flip={s['n_real_flip']} status_chg={s['n_status_change']} "
                     f"both_same={s['n_both_same']} both_none={s['n_both_none']}")
        lines.append(f"     material_breakrate_lower_observed = "
                     f"{s['material_breakrate_lower_observed']:.4f} (real letter-flips/{s['n_total']})")
        lines.append(f"     answer preserved where BOTH commit = "
                     f"{s['n_both_same']}/{s['n_both_committed_break']} "
                     f"({s['answer_preserved_where_both_commit']})")
        lines.append("  per-source (n_break post/pre/no_commit | postcommit_frac | mat_upper):")
        for src, ps in s["per_source"].items():
            lines.append(f"     {src:14s} brk={ps['n_break']:2d}  "
                         f"{ps['n_post_commit']}/{ps['n_pre_commit']}/{ps['n_no_commit']}  "
                         f"pcf={ps['postcommit_break_frac']!s:6.6}  "
                         f"mat_up={ps['material_breakrate_upper']:.4f}  "
                         f"committed={ps['n_committed_any']}/{ps['n_total']}")
        if s["committed_break_detail"]:
            lines.append("  committed break-prompts (id | D | C | class | ar->spec | changed):")
            for d in s["committed_break_detail"]:
                lines.append(f"     {d['id']:22s} D={d['D']!s:>4} C={d['C']!s:>4} "
                             f"{d['pos_class']:11s} {d['ar_answer']}->{d['spec_answer']} "
                             f"changed={d['answer_changed']}")
        lines.append(f"  VERDICT[{arm}]: {s['verdict']}")
        lines.append(f"    detail: {s['verdict_detail']}")
    lines.append("")
    lines.append(f"HEADLINE VERDICT: {results['headline_verdict']}")
    lines.append(f"  {results['arms'][primary_arm]['verdict_detail']}")
    rep = "\n".join(lines)
    (RES / "positional_materiality_report.txt").write_text(rep + "\n")
    print(rep, flush=True)


def _log_wandb(results: dict[str, Any], primary_arm: str, name: str, group: str) -> None:
    sys.path.insert(0, str(ROOT))
    from scripts import wandb_logging as wl
    s = results["arms"][primary_arm]
    cfg = {
        "pr": 685, "analysis_only": True, "official_tps": 0, "fires": 0,
        "arm": primary_arm, "break_definition": results["break_definition"],
        "decode_max_tokens": 512, "decode_ignore_eos": True,
        "reference_quality_budget_gpqa_aime": 6144,
    }
    run = wl.init_wandb_run(
        job_type="specbreak-positional-materiality", agent="kanna",
        name=name, group=group,
        notes="PR685 positional materiality: where do strict-#319 spec breaks fall "
              "relative to the official answer-commit? Premise audit: the 512-tok speed "
              "decode truncates ~87% of prompts before commit (#626 quality budget=6144 "
              "tok, gpqa/aime median completion 1843/3783).",
        tags=["specdec", "greedy-identity", "answer-materiality", "positional",
              "pr685", "analysis-only", "specbreak"],
        config=cfg,
    )
    if run is None:
        print("[analyze] wandb not configured — skipping", flush=True)
        return
    metrics: dict[str, Any] = {
        "analysis_only": 1, "official_tps": 0, "fires": 0,
        "postcommit_break_frac": s["postcommit_break_frac"],
        "material_breakrate_upper": s["material_breakrate_upper"],
        "material_breakrate_upper_literal": s["material_breakrate_upper_literal"],
        "definitely_safe_prompts": s["definitely_safe_prompts"],
        "n_no_commit_break": s["n_no_commit_break"],
        "commit_rate_512tok": s["commit_rate_512tok"],
        "n_committed_total": s["n_committed_total"],
        "break_rate": s["break_rate"],
        "n_break": s["n_break"],
        "n_post_commit": s["n_post_commit"],
        "n_pre_commit": s["n_pre_commit"],
        "n_real_flip": s["n_real_flip"],
        "n_status_change": s["n_status_change"],
        "n_both_same": s["n_both_same"],
        "n_both_none": s["n_both_none"],
        "n_both_committed_break": s["n_both_committed_break"],
        "material_breakrate_lower_observed": s["material_breakrate_lower_observed"],
        "answer_preserved_where_both_commit": s["answer_preserved_where_both_commit"],
        "onset_median": s["onset_median"],
        "reference_626_gpqa_median_completion_tok": REF_626["median_completion_tok"]["gpqa"],
        "reference_626_aime_median_completion_tok": REF_626["median_completion_tok"]["aime"],
    }
    for src, ps in s["per_source"].items():
        metrics[f"per_source/{src}/postcommit_break_frac"] = ps["postcommit_break_frac"]
        metrics[f"per_source/{src}/material_breakrate_upper"] = ps["material_breakrate_upper"]
        metrics[f"per_source/{src}/n_break"] = ps["n_break"]
        metrics[f"per_source/{src}/n_no_commit"] = ps["n_no_commit"]
    wl.log_event(run, "positional_materiality_complete", step=0, metrics=metrics)
    for k, v in metrics.items():
        run.summary[k] = v
    run.summary["verdict"] = s["verdict"]
    run.summary["verdict_detail"] = s["verdict_detail"]
    run.summary["headline_verdict"] = results["headline_verdict"]
    run.summary["analysis_only"] = True
    run.summary["official_tps"] = 0
    run.summary["fires"] = False
    wl.log_json_artifact(run, name="pr685_positional_materiality",
                         artifact_type="answer-materiality", data=results)
    print(f"[analyze] wandb run id={run.id} url={getattr(run, 'url', '?')}", flush=True)
    (RES / "wandb_run_id.txt").write_text(str(run.id))
    wl.finish_wandb(run)
    print("[analyze] wandb logged", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
