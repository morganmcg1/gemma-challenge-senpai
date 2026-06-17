#!/usr/bin/env python3
"""PR #619 -- dump full transcripts of McNemar-discordant (n01) items for manual read.

The PR asks to literally READ the prompt / base completion / int4 completion / correct
answer for the cells that drive the regression (base correct AND int4-body wrong). This
re-derives the n01 cells and their analyze.py category, then prints SAMPLES per category
so the automated tagging can be eyeballed for correctness.

  /tmp/land-inspect/bin/python dump_transcripts.py --cat i_truncated --n 3
  /tmp/land-inspect/bin/python dump_transcripts.py --cat iv_genuine  --n 6 --tail 1400
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

from inspect_ai.log import read_eval_log

P598 = Path("/workspace/senpai/target/research/validity/gpqa_larger_instrument_ci/results")
ANSWER_PAT = re.compile(r"ANSWER\s*[:=]\s*\(?([ABCD])\)?", re.I)


def load_cells(prefix):
    cell, meta, logs = {}, {}, {}
    for f in sorted(glob.glob(str(P598 / f"{prefix}_gpqa_main_mt8_s*.json"))):
        d = json.load(open(f))
        seed = d["sampling_seed"]
        logs[seed] = d["eval_log"]
        for r in d["per_sample"]:
            if r.get("value") not in ("C", "I"):
                continue
            cell[(r["id"], seed)] = bool(r["correct"])
            meta[(r["id"], seed)] = r
    return cell, meta, logs


def read_logs(logs):
    out = {}
    for seed, el in logs.items():
        log = read_eval_log(el)
        for s in (log.samples or []):
            o = getattr(s, "output", None)
            stop, text = None, ""
            prompt = ""
            # recover the user prompt text
            inp = getattr(s, "input", None)
            if isinstance(inp, str):
                prompt = inp
            else:
                msgs = getattr(s, "messages", None) or []
                for m in msgs:
                    if getattr(m, "role", None) == "user":
                        c = getattr(m, "content", "")
                        prompt = c if isinstance(c, str) else str(c)
            if o is not None:
                text = o.completion or ""
                ch = getattr(o, "choices", None)
                if ch:
                    stop = getattr(ch[0], "stop_reason", None)
                if stop is None:
                    stop = getattr(o, "stop_reason", None)
            out[(str(s.id), seed)] = {"stop": stop, "text": text, "prompt": prompt}
    return out


def categorize(n01, f_meta, f_st):
    cats = {"i_truncated": [], "ii_empty": [], "iii_extraction": [], "iv_genuine": []}
    for k in n01:
        fr = f_meta[k]
        stop = f_st[k]["stop"]
        text = f_st[k]["text"]
        ans = fr["answer"]
        if fr["empty"]:
            cats["ii_empty"].append(k)
        elif stop == "max_tokens":
            cats["i_truncated"].append(k)
        else:
            m = ANSWER_PAT.findall(text)
            stated = m[-1].upper() if m else None
            tgt = fr["target"]
            if stated is not None and stated == tgt and ans != tgt:
                cats["iii_extraction"].append(k)
            else:
                cats["iv_genuine"].append(k)
    return cats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cat", default="iv_genuine")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--tail", type=int, default=1200, help="chars of completion tail to show")
    ap.add_argument("--head", type=int, default=900, help="chars of prompt head to show")
    args = ap.parse_args()

    b_cell, b_meta, b_logs = load_cells("base")
    f_cell, f_meta, f_logs = load_cells("base_fullhead")
    f_st = read_logs(f_logs)
    b_st = read_logs(b_logs)

    shared = b_cell.keys() & f_cell.keys()
    n01 = [k for k in shared if b_cell[k] and not f_cell[k]]
    cats = categorize(n01, f_meta, f_st)
    print(f"n01={len(n01)}  " + "  ".join(f"{k}={len(v)}" for k, v in cats.items()))
    sel = cats[args.cat][: args.n]
    for k in sel:
        i, s = k
        fr, br = f_meta[k], b_meta[k]
        print("\n" + "=" * 100)
        print(f"id={i} seed={s} | target={fr['target']}  base_ans={br['answer']}(correct={br['correct']})  "
              f"int4_ans={fr['answer']}(correct={fr['correct']})")
        print(f"int4 stop={f_st[k]['stop']} chars={fr['completion_chars']} | "
              f"base stop={b_st[k]['stop']} chars={br['completion_chars']}")
        print("-" * 40 + " PROMPT (head) " + "-" * 40)
        print((f_st[k]["prompt"] or "")[: args.head])
        print("-" * 40 + " BASE completion (tail) " + "-" * 40)
        print((b_st[k]["text"] or "")[-args.tail:])
        print("-" * 40 + " INT4 completion (tail) " + "-" * 40)
        print((f_st[k]["text"] or "")[-args.tail:])


if __name__ == "__main__":
    main()
