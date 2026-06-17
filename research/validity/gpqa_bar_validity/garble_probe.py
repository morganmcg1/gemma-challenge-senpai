#!/usr/bin/env python3
"""PR #614 token-corruption probe. Sends a handful of GREEDY prompts whose correct
continuation is unambiguous, then flags fused/malformed-token corruption (the
"monopolesoles" / "wavelengthed" / "viewersight" signature seen on the len-6144 +
chunked-prefill serve). A clean serve must produce well-formed words; corruption
shows up as known-word mangling. Pure diagnostic; no scoring, no served-file change.
"""
import json
import re
import sys
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000/v1"
PROMPTS = [
    "Explain in 4 sentences why the sky is blue. Think step by step.",
    "List the first 10 chemical elements in order with their one- or two-letter symbols.",
    "Write Maxwell's four equations in differential form, naming each one.",
    "Count from one to fifteen, writing each number as a word.",
    "Define 'photosynthesis' in exactly two sentences.",
]
# Generic corruption signatures: a real word immediately glued to a suffix fragment,
# dropped intra-LaTeX spaces, tripled letters, and a small set of observed mangles.
CORRUPT_RE = [
    re.compile(r"\b\w*?([a-z]{3,})\1\w*\b"),          # internal stem duplication (monopol-es-oles-ish)
    re.compile(r"partial[a-z]"),                          # \partialt  (dropped space)
    re.compile(r"\b\w+ed\b(?=\s+(lights|ones)\b)"),     # wavelengthed lights
    re.compile(r"\b(?:viewersight|monopolesoles|Amptere|Amparat|Wiebach|Circurlsource|enstreamer|byatching)\b"),
    re.compile(r"([A-Za-z])\1\1"),                       # 3+ same letter in a row
]


def ask(prompt: str) -> str:
    body = json.dumps({
        "model": "gemma-4-e4b-it",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0, "top_p": 1.0, "max_tokens": 320, "seed": 0,
    }).encode()
    req = urllib.request.Request(BASE + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.load(r)
    return d["choices"][0]["message"]["content"]


def flags(text: str):
    hits = []
    for rx in CORRUPT_RE:
        for m in rx.finditer(text):
            w = m.group(0)
            if w.lower() in {"aaa", "iii"}:
                continue
            hits.append(w)
    return hits


def main():
    total_hits = 0
    for i, p in enumerate(PROMPTS, 1):
        out = ask(p)
        h = flags(out)
        total_hits += len(h)
        print(f"\n=== PROMPT {i}: {p}")
        print(out.strip())
        print(f"--- corruption signatures: {h if h else 'NONE'}")
    print(f"\n==== TOTAL corruption signatures across {len(PROMPTS)} prompts: {total_hits} ====")
    print("VERDICT:", "CORRUPT" if total_hits >= 2 else "CLEAN(ish)")


if __name__ == "__main__":
    main()
