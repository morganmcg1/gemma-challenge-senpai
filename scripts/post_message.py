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

from scripts.common import (
    DEFAULT_API,
    agent_id,
    hf,
    hf_bucket_uri,
    load_dotenv,
    post_json,
    run,
    safe_slug,
    utc_slug_stamp,
)


def body_from_args(args: argparse.Namespace) -> str:
    if args.body_file:
        return Path(args.body_file).read_text(encoding="utf-8").strip()
    if args.body:
        return args.body.strip()
    if not sys.stdin.isatty():
        return sys.stdin.read().strip()
    raise SystemExit("Provide --body, --body-file, or pipe markdown on stdin.")


def yaml_string(value: str) -> str:
    return json.dumps(value)


def bucket_markdown(args: argparse.Namespace, body: str) -> str:
    lines = ["---", f"type: {yaml_string(args.type)}"]
    if args.refs:
        lines.append(f"refs: {yaml_string(' '.join(args.refs))}")
    if args.priority:
        lines.append(f"priority: {yaml_string(args.priority)}")
    lines.extend(["---", "", body.rstrip(), ""])
    return "\n".join(lines)


def publish_bucket_message(args: argparse.Namespace, agent: str, body: str) -> dict:
    text = bucket_markdown(args, body)
    if args.dry_run:
        print(text, end="")
        return {}

    slug_basis = args.slug or body.splitlines()[0][:60]
    dest = hf_bucket_uri(agent, f"drafts/messages/{utc_slug_stamp()}-{safe_slug(slug_basis, default='message')}.md")
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".md", delete=False) as handle:
        handle.write(text)
        local_path = Path(handle.name)

    run(hf("buckets", "cp", str(local_path), dest))
    return post_json(f"{args.api}/v1/messages", {"source": dest})


def publish_raw_message(args: argparse.Namespace, agent: str, body: str) -> dict:
    payload = {"agent_id": agent, "body": body}
    if args.type:
        payload["type"] = args.type
    if args.refs:
        payload["refs"] = " ".join(args.refs)

    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return {}

    return post_json(f"{args.api}/v1/messages", payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Post a Gemma challenge board message.")
    parser.add_argument("--agent-id", default=None, help="Posting agent id. Defaults to AGENT_ID or senpai.")
    parser.add_argument("--api", default=DEFAULT_API)
    parser.add_argument("--mode", choices=["bucket", "raw"], default="bucket")
    parser.add_argument("--type", default="agent", choices=["agent", "system", "user"])
    parser.add_argument("--refs", action="append", default=[], help="Message/result filename to reference. May be repeated.")
    parser.add_argument("--priority", choices=["low", "normal", "high"], default="")
    parser.add_argument("--slug", default="", help="Scratch-bucket draft filename hint for bucket mode.")
    parser.add_argument("--body", default="")
    parser.add_argument("--body-file", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    load_dotenv()
    agent = agent_id(args.agent_id)
    body = body_from_args(args)
    response = (
        publish_raw_message(args, agent, body)
        if args.mode == "raw"
        else publish_bucket_message(args, agent, body)
    )
    if response:
        print(json.dumps(response, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
