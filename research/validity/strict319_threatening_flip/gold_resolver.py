"""GOLD for the FLIPPED #694 subset, loaded from the audited resolution.

GOLD[pid] = ("mc", LETTER) | ("num", NUMSTR), in THIS prompt's option lettering
(the format #689's extract_answer returns). Source of truth: gold_audit.json,
produced reproducibly by resolve_gold.py from the canonical gpqa_diamond /
MMLU-Pro datasets. Only prompts that actually flip (AR != spec, both committed)
are resolved -- PRESERVED prompts contribute 0 to R2W regardless of gold.
"""
from __future__ import annotations

import json
from pathlib import Path

_AUDIT = Path(__file__).resolve().parent / "gold_audit.json"

GOLD: dict[str, tuple[str, str]] = {}
if _AUDIT.exists():
    _data = json.loads(_AUDIT.read_text())
    for _pid, _a in _data.items():
        if _pid == "_meta":
            continue
        GOLD[_pid] = (_a["gold"][0], _a["gold"][1])
