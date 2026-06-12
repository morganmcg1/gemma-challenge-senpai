#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.common import HARNESS_SOURCE, ROOT, run


def main() -> None:
    dest = ROOT / "official" / "speed_benchmark"
    dest.parent.mkdir(parents=True, exist_ok=True)
    run(["hf", "buckets", "sync", HARNESS_SOURCE, str(dest)])
    print(f"Synced official harness to {dest}")


if __name__ == "__main__":
    main()
