#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.common import MAIN_BUCKET_README_SOURCE, OFFICIAL_MIRROR, SHARED_RESOURCES_SOURCE, hf, run


def remove_python_caches(root: Path) -> None:
    for pattern in ("*.pyc", "*.pyo"):
        for path in root.rglob(pattern):
            path.unlink()
    for path in sorted(root.rglob("__pycache__"), reverse=True):
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync stable files from the official Gemma central bucket.")
    parser.parse_args()

    OFFICIAL_MIRROR.mkdir(parents=True, exist_ok=True)
    shared_dest = OFFICIAL_MIRROR / "shared_resources"
    shared_dest.mkdir(parents=True, exist_ok=True)

    run(hf("buckets", "cp", MAIN_BUCKET_README_SOURCE, str(OFFICIAL_MIRROR / "README.md")))
    run(hf("buckets", "sync", SHARED_RESOURCES_SOURCE, str(shared_dest), "--delete"))
    remove_python_caches(OFFICIAL_MIRROR)

    print(f"Synced official central-bucket resources to {OFFICIAL_MIRROR}")


if __name__ == "__main__":
    main()
