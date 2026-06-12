from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_API = "https://gemma-challenge-gemma-bucket-sync.hf.space"
DEFAULT_AGENT_ID = "senpai"
MAIN_BUCKET_SOURCE = "hf://buckets/gemma-challenge/gemma-main-bucket"
MAIN_BUCKET_README_SOURCE = f"{MAIN_BUCKET_SOURCE}/README.md"
SHARED_RESOURCES_SOURCE = f"{MAIN_BUCKET_SOURCE}/shared_resources/"
OFFICIAL_MIRROR = ROOT / "official" / "main_bucket"
SPEED_BENCHMARK_DIR = OFFICIAL_MIRROR / "shared_resources" / "speed_benchmark"
HARNESS_SOURCE = f"{SHARED_RESOURCES_SOURCE}speed_benchmark/"


def load_dotenv(path: Path = ROOT / ".env") -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def agent_id(value: str | None = None) -> str:
    return value or os.environ.get("AGENT_ID") or DEFAULT_AGENT_ID


def scratch_bucket(agent: str) -> str:
    return f"gemma-challenge/gemma-{agent}"


def hf_bucket_uri(agent: str, prefix: str) -> str:
    return f"hf://buckets/{scratch_bucket(agent)}/{prefix.strip('/')}"


def submission_name(path: Path, value: str | None = None) -> str:
    return value or path.name.replace("_", "-")


def submission_prefix(agent: str, name: str) -> str:
    return f"submissions/{agent}/{name}"


def run_prefix(agent: str, name: str) -> str:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return f"results/{agent}/{name}-{stamp}"


def run(command: list[str], *, cwd: Path = ROOT, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        capture_output=capture,
    )


def hf(*args: str) -> list[str]:
    binary = shutil.which("hf")
    if binary:
        return [binary, *args]
    return ["uv", "run", "hf", *args]


def hf_json(uri: str) -> dict[str, Any]:
    result = run(hf("buckets", "cp", uri, "-"), capture=True)
    return json.loads(result.stdout)


def post_json(url: str, payload: dict[str, Any], *, token: str | None = None) -> dict[str, Any]:
    headers = {"content-type": "application/json"}
    if token:
        headers["authorization"] = f"Bearer {token}"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{url} returned HTTP {exc.code}: {body}") from exc
    return json.loads(body) if body else {}


def require_hf_token() -> str:
    load_dotenv()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is required in the environment or .env")
    return token
