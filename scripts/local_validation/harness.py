"""Serve a submission locally and run the hardware-independent checks.

This mirrors what the official ``hf_bucket_single_job.py`` does inside an HF Job,
minus the bucket plumbing: create a participant venv from ``manifest.json``
dependencies, start ``serve.py`` with the manifest env, wait for ``/v1/models``,
then drive the official ``decode_outputs.py`` / ``ppl_endpoint.py`` against the
live endpoint. TPS here is an *exploratory* single-stream probe on the A10G, not
the official a10g-small score.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from . import paths

VENV_ROOT = Path("/tmp/senpai-venvs")


def _run(cmd: list[str]) -> None:
    print("    $", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def load_manifest(submission_dir: Path, manifest_name: str = "manifest.json") -> dict[str, Any]:
    path = submission_dir / manifest_name
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"manifest must be a JSON object: {path}")
    if not data.get("serve") or not isinstance(data["serve"], list):
        raise ValueError("manifest must define a non-empty 'serve' command list")
    deps = data.get("dependencies") or []
    if not isinstance(deps, list) or not all(isinstance(x, str) for x in deps):
        raise ValueError("manifest 'dependencies' must be a list of strings")
    data["dependencies"] = deps
    return data


def ensure_server_venv(dependencies: list[str], python: str = "3.12") -> Path:
    """Create (or reuse) a venv keyed by its dependency set; return its python.

    Keying on the dependency hash lets repeat validations of the same submission
    skip the (slow) vLLM install entirely.
    """
    uv = shutil.which("uv")
    if not uv:
        raise RuntimeError("uv is required to build the server venv")
    key = hashlib.sha256(("\n".join([python, *sorted(dependencies)])).encode()).hexdigest()[:16]
    venv = VENV_ROOT / key
    py = venv / "bin" / "python"
    marker = venv / ".deps_installed"
    if py.exists() and marker.exists():
        return py
    VENV_ROOT.mkdir(parents=True, exist_ok=True)
    _run([uv, "venv", str(venv), "--python", python])
    if dependencies:
        _run([uv, "pip", "install", "--python", str(py), *dependencies])
    marker.write_text("\n".join(sorted(dependencies)))
    return py


def resolve_model_id(value: str, submission_dir: Path) -> str:
    p = Path(value)
    if p.is_absolute():
        return str(p)
    candidate = submission_dir / value
    return str(candidate) if candidate.exists() else value


def serve_model_id(manifest: dict[str, Any], submission_dir: Path) -> str:
    """The ``MODEL_ID`` value serve.py will actually receive for this submission.

    Mirrors [[_participant_env]] exactly so callers (e.g. the reference
    generator) can recover the served identity without standing up a server: an
    explicit manifest ``env.MODEL_ID`` wins, then an ambient ``MODEL_ID`` already
    in the process env, else the manifest ``model_id`` resolved against the
    submission dir.
    """
    env_block = manifest.get("env") or {}
    if "MODEL_ID" in env_block:
        return str(env_block["MODEL_ID"])
    if "MODEL_ID" in os.environ:
        return os.environ["MODEL_ID"]
    return resolve_model_id(str(manifest.get("model_id", paths.BF16_MODEL)), submission_dir)


def reference_identity(model_id: str, submission_dir: Path | None) -> str:
    """Canonical, collision-free identity used to KEY a greedy reference artifact.

    Two invariants must hold or the gate silently mis-resolves: (1) the generator
    must write a submission's reference to the same tag ``validate_submission``
    reads it from, and (2) two *distinct* submissions must never share a tag —
    even when their manifests declare the same ``model_id``. The served
    ``MODEL_ID`` is not safe to key on: several int4 submissions all set
    ``env.MODEL_ID="model"`` (a relative literal), a bucket-weights submission can
    nominally report the bf16 hub id while serving entirely different baked
    weights, and two submissions could point at one shared external checkpoint.
    Keying on the model id alone would alias any of these onto one reference.

    So a *submission's* reference is anchored to its (absolute) directory:
    ``<submission_dir>::<model_id>``. The dir is unique per submission, so two
    submissions can never collide regardless of what ``model_id`` they declare;
    and because a bundled checkpoint lives at ``<submission_dir>/model``, anchoring
    to the dir yields the same tag as keying by that absolute checkpoint path
    (``::`` and ``/`` both normalize to ``__`` in [[paths.model_tag]]) while also
    covering hub-id and shared-external-checkpoint submissions. Only a bare
    baseline reference (``submission_dir is None``, e.g. ``gen_greedy_reference
    --model-id``) keys purely by model id — the shared plain-checkpoint anchor,
    intentionally not tied to any one submission.
    """
    if submission_dir is None:
        return model_id
    return f"{Path(submission_dir).resolve()}::{model_id}"


def assert_submission_reference_tag(ref_tag: str) -> str:
    """Fail loudly if a SUBMISSION's resolved reference tag collapses to a bare model id.

    PR #32 anchored a submission's greedy reference to ``<submission_dir>::<model_id>``
    precisely so several int4 submissions that all set ``env.MODEL_ID="model"`` (a
    relative literal) could not silently alias onto one ``research/greedy_reference/model/``
    reference — a confident *wrong* GREEDY_IDENTICAL/DIVERGENT. This guard pins that
    fix at runtime: a submission tag must carry the ``::`` directory anchor and never be
    the bare literal ``"model"``. It catches a regression to bare-``model_id`` keying or
    an empty ``submission_dir`` collapsing the anchor. Baseline references
    (``submission_dir is None``, keyed purely by model id via [[reference_identity]]) are
    intentionally bare and must NOT be passed through this guard.
    """
    assert ref_tag != "model" and "::" in ref_tag, (
        f"Reference tag {ref_tag!r} looks like a bare model_id — "
        "collision risk; check submission_dir and model_id are both non-empty"
    )
    return ref_tag


def _participant_env(
    manifest: dict[str, Any], submission_dir: Path, server_venv: Path, port: int
) -> dict[str, str]:
    env = os.environ.copy()
    for k, v in (manifest.get("env") or {}).items():
        env[str(k)] = str(v)
    env["VIRTUAL_ENV"] = str(server_venv)
    env["PATH"] = f"{server_venv / 'bin'}{os.pathsep}{env.get('PATH', '')}"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.setdefault("MODEL_ID", serve_model_id(manifest, submission_dir))
    env.setdefault(
        "SERVED_MODEL_NAME", str(manifest.get("served_model_name", paths.DEFAULT_SERVED_NAME))
    )
    env.setdefault("HOST", "127.0.0.1")
    env["PORT"] = str(port)
    return env


def _build_serve_command(command: list[str], submission_dir: Path, server_python: Path) -> list[str]:
    cmd = list(command)
    if cmd[0] in {"python", "python3"}:
        cmd[0] = str(server_python)
    elif cmd[0].endswith(".py"):
        script = Path(cmd[0])
        if not script.is_absolute():
            script = submission_dir / script
        cmd = [str(server_python), str(script), *cmd[1:]]
    return cmd


def _terminate(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:
        proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            proc.kill()


class LocalServer:
    """Context manager that serves a submission's endpoint on localhost."""

    def __init__(
        self,
        submission_dir: Path,
        *,
        server_python: Path,
        port: int = 8000,
        startup_timeout_s: int = 1200,
        manifest_name: str = "manifest.json",
        log_path: Path | None = None,
        extra_env: dict[str, str] | None = None,
    ) -> None:
        self.submission_dir = Path(submission_dir)
        self.server_python = Path(server_python)
        self.port = port
        self.startup_timeout_s = startup_timeout_s
        self.manifest = load_manifest(self.submission_dir, manifest_name)
        self.base_url = f"http://127.0.0.1:{port}"
        self.log_path = log_path
        self.extra_env = extra_env or {}
        self.proc: subprocess.Popen | None = None
        env = _participant_env(self.manifest, self.submission_dir, self.server_python.parent.parent, port)
        env.update(self.extra_env)
        self.env = env
        self.served_model_name = env["SERVED_MODEL_NAME"]
        self.model_id = env["MODEL_ID"]
        # The identity serve.py consumes (``model_id``) is NOT safe to key the
        # greedy reference on — see [[reference_identity]]. Resolve a separate,
        # submission-anchored identity for reference lookup; the served path is
        # untouched.
        self.reference_model_id = reference_identity(self.model_id, self.submission_dir)
        # A LocalServer always serves a concrete submission, so its reference tag
        # must be the collision-free <submission_dir>::<model_id> form — never the
        # bare 'model' literal that pre-#32 keying produced (see [[reference_identity]]).
        assert_submission_reference_tag(self.reference_model_id)

    def _wait_ready(self) -> None:
        deadline = time.time() + self.startup_timeout_s
        last = ""
        while time.time() < deadline:
            if self.proc is not None and self.proc.poll() is not None:
                raise RuntimeError(
                    f"server exited before readiness with code {self.proc.returncode} "
                    f"(see {self.log_path})"
                )
            try:
                with urllib.request.urlopen(f"{self.base_url}/v1/models", timeout=5.0) as r:
                    if r.status == 200:
                        return
                    last = f"status={r.status}"
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last = str(exc)
            time.sleep(5)
        raise RuntimeError(f"endpoint not ready at {self.base_url}/v1/models: {last}")

    def __enter__(self) -> "LocalServer":
        serve_cmd = _build_serve_command(self.manifest["serve"], self.submission_dir, self.server_python)
        print(f"[serve] {' '.join(serve_cmd)}", flush=True)
        print(f"[serve] MODEL_ID={self.model_id} env={ {k: self.env[k] for k in self.manifest.get('env', {})} }", flush=True)
        log = open(self.log_path, "w") if self.log_path else subprocess.DEVNULL
        self._log_handle = log
        t0 = time.time()
        self.proc = subprocess.Popen(
            serve_cmd,
            cwd=self.submission_dir,
            env=self.env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            preexec_fn=os.setsid,
        )
        self._wait_ready()
        print(f"[serve] ready at {self.base_url} in {time.time() - t0:.0f}s", flush=True)
        return self

    def __exit__(self, *exc) -> None:
        _terminate(self.proc)
        if getattr(self, "_log_handle", None) not in (None, subprocess.DEVNULL):
            try:
                self._log_handle.close()
            except Exception:
                pass


def capture_decode(
    runner_python: Path,
    *,
    base_url: str,
    model: str,
    out_file: Path,
    summary_file: Path,
    num_prompts: int = paths.NUM_PROMPTS,
    output_len: int = paths.OUTPUT_LEN,
    seed: int = paths.SEED,
    tokenizer: str = paths.TOKENIZER,
    dataset: Path | None = None,
    timeout_s: int = 3600,
) -> dict[str, Any]:
    """Run the official decode_outputs.py to capture token IDs from the endpoint."""
    dataset = dataset or paths.EVAL_PROMPTS
    out_file.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(runner_python), str(paths.DECODE_SCRIPT),
        "--base-url", base_url.rstrip("/"),
        "--model", model,
        "--dataset-path", str(dataset),
        "--output-file", str(out_file),
        "--summary-file", str(summary_file),
        "--tokenizer", tokenizer,
        "--num-prompts", str(num_prompts),
        "--output-len", str(output_len),
        "--seed", str(seed),
    ]
    print("[decode]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, timeout=timeout_s)
    return json.loads(summary_file.read_text())


def run_ppl(
    runner_python: Path,
    *,
    base_url: str,
    model: str,
    out_file: Path,
    summary_file: Path,
    dataset: Path | None = None,
    timeout_s: int = 1800,
) -> dict[str, Any]:
    """Run the official ppl_endpoint.py against the ground-truth tokens."""
    dataset = dataset or paths.ppl_dataset()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(runner_python), str(paths.PPL_SCRIPT),
        "--base-url", base_url.rstrip("/"),
        "--model", model,
        "--dataset-path", str(dataset),
        "--output-file", str(out_file),
        "--summary-file", str(summary_file),
    ]
    print("[ppl]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, timeout=timeout_s)
    return json.loads(summary_file.read_text())


def _completion(base_url: str, model: str, prompt: str, max_tokens: int, timeout_s: int = 300) -> dict[str, Any]:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
        "ignore_eos": True,
    }
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        resp = json.loads(r.read().decode())
    resp["_wall_s"] = time.time() - t0
    return resp


def probe_tps(
    base_url: str,
    model: str,
    *,
    decode_tokens: int = 256,
    prompt: str = "Explain step by step how a transformer decodes one token at a time.",
) -> dict[str, Any]:
    """Exploratory single-stream decode TPS via two timed /v1/completions calls.

    Times a 1-token request (≈ prefill + 1 decode) and an N-token request, then
    isolates steady-state decode throughput as (N-1)/(wall_N - wall_1). This is a
    local A10G probe, NOT the official a10g-small score.
    """
    _completion(base_url, model, prompt, 8)  # warmup
    r1 = _completion(base_url, model, prompt, 1)
    rN = _completion(base_url, model, prompt, decode_tokens)
    wall1, wallN = r1["_wall_s"], rN["_wall_s"]

    def n_out(resp: dict[str, Any]) -> int:
        usage = resp.get("usage") or {}
        if isinstance(usage.get("completion_tokens"), int):
            return usage["completion_tokens"]
        ch = (resp.get("choices") or [{}])[0]
        tok = ch.get("token_ids")
        return len(tok) if isinstance(tok, list) else 0

    n = n_out(rN) or decode_tokens
    decode_tps = (n - 1) / (wallN - wall1) if wallN > wall1 else float("nan")
    return {
        "decode_tps_single_stream": decode_tps,
        "naive_tps": n / wallN if wallN else float("nan"),
        "ttft_s_approx": wall1,
        "decode_tokens": n,
        "wall_1tok_s": wall1,
        "wall_ntok_s": wallN,
        "note": "exploratory A10G single-stream probe; not the official a10g-small TPS",
    }
