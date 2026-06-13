#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.common import DEFAULT_AGENT_ID, DEFAULT_API


DEFAULT_STATE_DIR = ROOT_DIR / "research" / "frontier_watch"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def fetch_json(url: str) -> Any:
    request = urllib.request.Request(url, headers={"accept": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def endpoint(api: str, path: str, params: dict[str, Any] | None = None) -> str:
    url = f"{api.rstrip('/')}{path}"
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    return url


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f"{path.suffix}.tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def leaderboard_rows(payload: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    rows = payload.get("rows") or payload.get("leaderboard") or []
    return rows[:limit]


def item_names(payload: dict[str, Any]) -> list[str]:
    items = payload.get("items") or []
    names: list[str] = []
    for item in items:
        if isinstance(item, str):
            names.append(item)
        elif isinstance(item, dict):
            name = item.get("filename") or item.get("name")
            if name:
                names.append(str(name))
    return names


def message_names(digest: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for item in digest.get("recent_messages") or []:
        if isinstance(item, str):
            names.append(item)
        elif isinstance(item, dict) and item.get("filename"):
            names.append(str(item["filename"]))
    return names


def inbox_items(digest: dict[str, Any]) -> list[Any]:
    for key in ("inbox", "inbox_mentions", "mentions", "recent_mentions"):
        value = digest.get(key)
        if isinstance(value, list):
            return value
    return []


def tps(row: dict[str, Any]) -> float:
    value = row.get("tps") or row.get("output_tps") or 0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def fmt_tps(value: float) -> str:
    return f"{value:.2f}" if value else "n/a"


def row_key(row: dict[str, Any]) -> str:
    return str(row.get("filename") or f"{row.get('agent')}|{row.get('method')}|{row.get('timestamp')}")


def short(value: Any, limit: int = 90) -> str:
    text = str(value or "").replace("\n", " ")
    return text if len(text) <= limit else f"{text[: limit - 1]}..."


def collect_snapshot(api: str, agent: str, limit: int, top_k: int) -> dict[str, Any]:
    leaderboard = fetch_json(endpoint(api, "/v1/leaderboard"))
    digest = fetch_json(endpoint(api, "/v1/digest", {"as": agent}))
    results = fetch_json(endpoint(api, "/v1/results", {"limit": limit}))
    messages = fetch_json(endpoint(api, "/v1/messages", {"limit": limit}))
    taskforces = fetch_json(endpoint(api, "/v1/taskforces", {"limit": 20}))

    return {
        "fetched_at": utc_now(),
        "api": api,
        "agent": agent,
        "leaderboard_top": leaderboard_rows(leaderboard, top_k),
        "result_files": item_names(results),
        "message_files": item_names(messages),
        "digest_message_files": message_names(digest),
        "inbox": inbox_items(digest),
        "taskforces": taskforces.get("items") or [],
        "agent_count": (digest.get("agents") or {}).get("count"),
        "raw_digest_counts": {
            "messages": len(digest.get("recent_messages") or []),
            "leaderboard": len(digest.get("leaderboard") or []),
            "taskforces": (digest.get("taskforces") or {}).get("count"),
        },
    }


def compare(old: dict[str, Any] | None, new: dict[str, Any], min_tps_delta: float) -> dict[str, Any]:
    if not old:
        return {
            "first_snapshot": True,
            "frontier_changed": True,
            "new_result_files": new["result_files"][:10],
            "new_message_files": new["message_files"][:10],
            "top_delta_tps": 0.0,
        }

    old_results = set(old.get("result_files") or [])
    old_messages = set(old.get("message_files") or [])
    new_results = [name for name in new["result_files"] if name not in old_results]
    new_messages = [name for name in new["message_files"] if name not in old_messages]

    old_top = (old.get("leaderboard_top") or [{}])[0]
    new_top = (new.get("leaderboard_top") or [{}])[0]
    old_top_key = row_key(old_top)
    new_top_key = row_key(new_top)
    top_delta = tps(new_top) - tps(old_top)
    top_changed = old_top_key != new_top_key

    old_ranked = {row_key(row): row for row in old.get("leaderboard_top") or []}
    rank_changes: list[dict[str, Any]] = []
    for row in new.get("leaderboard_top") or []:
        previous = old_ranked.get(row_key(row))
        if not previous:
            rank_changes.append({"kind": "new_top_row", "row": row})
            continue
        if row.get("rank") != previous.get("rank") or abs(tps(row) - tps(previous)) >= min_tps_delta:
            rank_changes.append({"kind": "changed_top_row", "old": previous, "row": row})

    return {
        "first_snapshot": False,
        "frontier_changed": bool(
            top_changed
            or abs(top_delta) >= min_tps_delta
            or new_results
            or new.get("inbox")
        ),
        "top_changed": top_changed,
        "top_delta_tps": top_delta,
        "new_result_files": new_results[:15],
        "new_message_files": new_messages[:15],
        "rank_changes": rank_changes[:15],
    }


def render_markdown(snapshot: dict[str, Any], diff: dict[str, Any]) -> str:
    top_rows = snapshot.get("leaderboard_top") or []
    current_top = top_rows[0] if top_rows else {}
    inbox = snapshot.get("inbox") or []

    lines = [
        f"# Frontier Watch - {snapshot['fetched_at']}",
        "",
        f"- agent: `{snapshot['agent']}`",
        f"- agents registered: `{snapshot.get('agent_count') or 'unknown'}`",
        f"- frontier changed since previous snapshot: `{str(diff.get('frontier_changed')).lower()}`",
        f"- current #1: rank `{current_top.get('rank', 'n/a')}` `{current_top.get('agent', 'unknown')}` "
        f"{fmt_tps(tps(current_top))} TPS, verification `{current_top.get('verification', 'unknown')}`, "
        f"result `{current_top.get('filename', 'unknown')}`",
    ]

    if diff.get("top_changed"):
        lines.append(f"- #1 changed; TPS delta vs previous snapshot: `{diff.get('top_delta_tps', 0.0):+.2f}`")
    elif not diff.get("first_snapshot"):
        lines.append(f"- #1 unchanged; TPS delta vs previous snapshot: `{diff.get('top_delta_tps', 0.0):+.2f}`")

    if inbox:
        lines.append(f"- inbox mentions for advisor to inspect first: `{len(inbox)}`")

    lines.extend(["", "## Top Leaderboard", "", "| rank | agent | TPS | verification | method | result |", "|---:|---|---:|---|---|---|"])
    for row in top_rows:
        lines.append(
            f"| {row.get('rank', '')} | {row.get('agent', '')} | {fmt_tps(tps(row))} | "
            f"{row.get('verification', '')} | {short(row.get('method'), 45)} | {row.get('filename', '')} |"
        )

    lines.extend(["", "## New Since Previous Snapshot"])
    new_results = diff.get("new_result_files") or []
    new_messages = diff.get("new_message_files") or []
    rank_changes = diff.get("rank_changes") or []
    if new_results:
        lines.extend(["", "New result files:"])
        lines.extend(f"- `{name}`" for name in new_results)
    if new_messages:
        lines.extend(["", "New message files:"])
        lines.extend(f"- `{name}`" for name in new_messages)
    if rank_changes:
        lines.extend(["", "Top-rank changes:"])
        for change in rank_changes:
            row = change.get("row") or {}
            lines.append(
                f"- `{change.get('kind')}` rank {row.get('rank')}: `{row.get('agent')}` "
                f"{fmt_tps(tps(row))} TPS via `{short(row.get('method'), 60)}`"
            )
    if not any((new_results, new_messages, rank_changes)):
        lines.append("")
        lines.append("No new result/message/rank movement detected in the configured window.")

    if snapshot.get("taskforces"):
        lines.extend(["", "## Active Taskforces"])
        for taskforce in snapshot["taskforces"]:
            lines.append(
                f"- `{taskforce.get('name')}`: last activity `{taskforce.get('last_activity', 'unknown')}`, "
                f"contributors `{', '.join(taskforce.get('contributors') or [])}`"
            )

    lines.extend(
        [
            "",
            "## Advisor Instructions",
            "",
            "- Before assigning or reviewing new work, inspect any new top result/result file listed above.",
            "- If the #1 result changed, update active hypotheses that cite an older frontier baseline.",
            "- If new verifier, human, negative-result, or taskforce messages appeared, read those before spending HF Jobs quota.",
            "- Cite the relevant result/message/taskforce filename in any new PR or issue.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Poll the live Gemma challenge frontier and write advisor-ready context.")
    parser.add_argument("--agent-id", default=DEFAULT_AGENT_ID)
    parser.add_argument("--api", default=DEFAULT_API)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument("--limit", type=int, default=80, help="Number of recent message/result filenames to compare.")
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--min-tps-delta", type=float, default=1.0)
    parser.add_argument(
        "--exit-code-on-change",
        action="store_true",
        help="Exit 2 when new frontier context should be injected into the advisor.",
    )
    args = parser.parse_args()

    latest_path = args.state_dir / "latest.json"
    previous_path = args.state_dir / "previous.json"
    markdown_path = args.state_dir / "latest.md"

    old = read_json(latest_path)
    snapshot = collect_snapshot(args.api, args.agent_id, args.limit, args.top_k)
    diff = compare(old, snapshot, args.min_tps_delta)
    snapshot["diff"] = diff
    markdown = render_markdown(snapshot, diff)

    args.state_dir.mkdir(parents=True, exist_ok=True)
    if old:
        write_json(previous_path, old)
    write_json(latest_path, snapshot)
    markdown_path.write_text(markdown, encoding="utf-8")
    print(markdown, end="")

    if args.exit_code_on_change and diff.get("frontier_changed"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
