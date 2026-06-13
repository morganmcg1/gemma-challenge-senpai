#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.common import DEFAULT_API, ROOT, agent_id, board_agent_id, get_json, load_dotenv


DEFAULT_STATE = ROOT / ".scratch" / "message-poll-state.json"


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"seen_inbox": [], "seen_mentions": [], "seen_messages": []}
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def seen_set(state: dict[str, Any], key: str) -> set[str]:
    return set(state.get(key) or [])


def remember(state: dict[str, Any], key: str, names: list[str], *, keep: int = 500) -> None:
    merged = list(dict.fromkeys([*(state.get(key) or []), *names]))
    state[key] = merged[-keep:]


def filename(item: dict[str, Any]) -> str:
    return str(item.get("filename") or item.get("name") or "")


def frontmatter_agent(item: dict[str, Any]) -> str:
    frontmatter = item.get("frontmatter")
    if isinstance(frontmatter, dict):
        return str(frontmatter.get("agent") or "")
    return str(item.get("agent") or "")


def body_text(item: dict[str, Any]) -> str:
    body = item.get("body") or item.get("snippet") or ""
    return str(body).strip()


def compact(text: str, *, limit: int = 420) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else f"{text[: limit - 1].rstrip()}..."


def digest(api: str, handle: str) -> dict[str, Any]:
    return get_json(f"{api}/v1/digest", {"as": handle})


def inbox_items(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    inbox = snapshot.get("inbox") or {}
    items = inbox.get("items") if isinstance(inbox, dict) else []
    return [item for item in items or [] if isinstance(item, dict)]


def recent_messages(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    return [item for item in snapshot.get("recent_messages") or [] if isinstance(item, dict)]


def mention_matches(messages: list[dict[str, Any]], handle: str) -> list[dict[str, Any]]:
    needle = f"@{handle.lower()}"
    return [item for item in messages if needle in body_text(item).lower()]


def render_items(title: str, items: list[dict[str, Any]]) -> None:
    if not items:
        return
    print(title)
    for item in items:
        name = filename(item)
        agent = frontmatter_agent(item) or "unknown"
        print(f"- {name} from {agent}: {compact(body_text(item))}")


def render_summary(snapshot: dict[str, Any], as_agent: str, handle: str, new_inbox: list[dict[str, Any]], new_mentions: list[dict[str, Any]]) -> None:
    generated_at = snapshot.get("generated_at", "")
    leaderboard = snapshot.get("leaderboard") or []
    top = leaderboard[0] if leaderboard else {}
    top_line = ""
    if top:
        top_line = f" top={top.get('agent')} {float(top.get('tps') or 0):.2f} TPS"
    print(f"board digest as @{as_agent}, watching @{handle}: inbox={len(inbox_items(snapshot))} new_inbox={len(new_inbox)} new_mentions={len(new_mentions)}{top_line} generated_at={generated_at}")
    render_items("new inbox", new_inbox)
    render_items("new direct @ mentions in recent messages", new_mentions)


def poll_once(args: argparse.Namespace, state: dict[str, Any]) -> bool:
    snapshot = digest(args.api, args.as_agent)
    inbox = inbox_items(snapshot)
    mentions = mention_matches(recent_messages(snapshot), args.handle)

    seen_inbox = seen_set(state, "seen_inbox")
    seen_mentions = seen_set(state, "seen_mentions")
    new_inbox = [item for item in inbox if filename(item) and filename(item) not in seen_inbox]
    new_mentions = [item for item in mentions if filename(item) and filename(item) not in seen_mentions]

    if args.json:
        print(json.dumps({"digest": snapshot, "new_inbox": new_inbox, "new_mentions": new_mentions}, indent=2, sort_keys=True))
    elif args.all or new_inbox or new_mentions:
        render_summary(snapshot, args.as_agent, args.handle, new_inbox, new_mentions)

    if args.mark_seen:
        remember(state, "seen_inbox", [filename(item) for item in inbox if filename(item)])
        remember(state, "seen_mentions", [filename(item) for item in mentions if filename(item)])
        remember(state, "seen_messages", [filename(item) for item in recent_messages(snapshot) if filename(item)])

    return bool(new_inbox or new_mentions)


def main() -> None:
    parser = argparse.ArgumentParser(description="Poll the Gemma challenge board digest and inbox.")
    parser.add_argument("--handle", default=None, help="Board mention handle to watch, without @. Defaults to BOARD_AGENT_ID, AGENT_ID, or senpai.")
    parser.add_argument("--as-agent", "--agent-id", dest="as_agent", default=None, help="Registered agent id used for digest/inbox. Defaults to AGENT_ID or senpai.")
    parser.add_argument("--api", default=DEFAULT_API)
    parser.add_argument("--state-file", default=str(DEFAULT_STATE))
    parser.add_argument("--interval", type=float, default=60.0)
    parser.add_argument("--watch", action="store_true", help="Keep polling and print new inbox/mention items.")
    parser.add_argument("--json", action="store_true", help="Print raw digest plus computed new items.")
    parser.add_argument("--all", action="store_true", help="Print a summary even when there are no new items.")
    parser.add_argument("--mark-seen", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    load_dotenv()
    args.handle = board_agent_id(args.handle)
    args.as_agent = agent_id(args.as_agent)
    state_path = Path(args.state_file)
    state = load_state(state_path)

    while True:
        poll_once(args, state)
        if args.mark_seen:
            save_state(state_path, state)
        if not args.watch:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
