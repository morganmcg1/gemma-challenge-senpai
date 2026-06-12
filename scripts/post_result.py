#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.common import DEFAULT_API, agent_id, hf, hf_bucket_uri, load_dotenv, post_json, require_hf_token, run, submission_prefix


def load_summary(path_or_uri: str) -> dict:
    if path_or_uri.startswith("hf://"):
        result = run(hf("buckets", "cp", path_or_uri, "-"), capture=True)
        return json.loads(result.stdout)
    return json.loads(Path(path_or_uri).read_text(encoding="utf-8"))


def result_markdown(args: argparse.Namespace, summary: dict, agent: str) -> str:
    tps = summary.get("tps") or summary.get("output_tps") or 0
    ppl = summary.get("ppl") or 0
    submission_uri = hf_bucket_uri(agent, submission_prefix(agent, args.submission_name))
    description = args.description or f"{args.method}: {tps:.2f} TPS / PPL {ppl}"
    body = args.body or "Result posted from gemma-challenge-senpai helper."
    return (
        "---\n"
        f"tps: {tps}\n"
        f"ppl: {ppl}\n"
        f"method: {args.method}\n"
        f"status: {args.status}\n"
        f"description: {description}\n"
        f"submission: {submission_uri}\n"
        "---\n\n"
        f"{body}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Draft or publish a Gemma challenge result file.")
    parser.add_argument("--agent-id", default=None)
    parser.add_argument("--summary", required=True, help="Local summary.json path or hf:// URI.")
    parser.add_argument("--method", required=True)
    parser.add_argument("--submission-name", required=True)
    parser.add_argument("--status", choices=["agent-run", "negative"], default="agent-run")
    parser.add_argument("--description", default="")
    parser.add_argument("--body", default="")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--api", default=DEFAULT_API)
    args = parser.parse_args()

    load_dotenv()
    agent = agent_id(args.agent_id)
    summary = load_summary(args.summary)
    text = result_markdown(args, summary, agent)
    safe_method = "".join(char if char.isalnum() or char in "._-" else "-" for char in args.method.lower())

    if not args.publish:
        print(text)
        return

    token = require_hf_token()
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".md", delete=False) as handle:
        handle.write(text)
        local_path = handle.name

    source = hf_bucket_uri(agent, f"results/{safe_method}.md")
    run(hf("buckets", "cp", local_path, source))
    response = post_json(f"{args.api}/v1/results", {"source": source}, token=token)
    print(response)


if __name__ == "__main__":
    main()
