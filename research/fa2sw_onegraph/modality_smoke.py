#!/usr/bin/env python
"""Modality + readiness smoke for a running fa2sw_onegraph endpoint.

Assumes serve.py is already serving on BASE_URL. Checks:
  1. /v1/models lists the served model (readiness),
  2. a text chat completion succeeds (greedy),
  3. an image chat completion succeeds (vision tower path is live),
  4. an audio chat completion succeeds (audio tower path is live).

A 200 with a non-empty completion is the success bar for modalities; we are
verifying the multimodal paths are wired and don't error, not scoring quality.
"""
from __future__ import annotations

import base64
import io
import json
import os
import struct
import sys
import urllib.request
import wave

BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:8000").rstrip("/")
MODEL = os.environ.get("SERVED_MODEL_NAME", "gemma-4-e4b-it")


def _post(path: str, payload: dict, timeout: int = 120) -> dict:
    req = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _get(path: str, timeout: int = 30) -> dict:
    with urllib.request.urlopen(f"{BASE_URL}{path}", timeout=timeout) as r:
        return json.loads(r.read().decode())


def _png_data_uri() -> str:
    from PIL import Image

    img = Image.new("RGB", (64, 64), (30, 120, 220))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _wav_data_uri() -> str:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        frames = b"".join(struct.pack("<h", 0) for _ in range(16000))  # 1s silence
        w.writeframes(frames)
    return "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode()


def _chat(content) -> dict:
    return _post(
        "/v1/chat/completions",
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 16,
            "temperature": 0.0,
        },
    )


def main() -> int:
    results = {}

    models = _get("/v1/models")
    ids = [m.get("id") for m in models.get("data", [])]
    results["models_ready"] = MODEL in ids
    print(f"[modality] /v1/models -> {ids}; ready={results['models_ready']}", flush=True)

    txt = _chat("In one word, what is the capital of France?")
    out = txt["choices"][0]["message"]["content"]
    results["text"] = bool(out)
    print(f"[modality] text -> {out!r}", flush=True)

    try:
        img = _chat([
            {"type": "text", "text": "What is the dominant color? One word."},
            {"type": "image_url", "image_url": {"url": _png_data_uri()}},
        ])
        out = img["choices"][0]["message"]["content"]
        results["image"] = bool(out)
        print(f"[modality] image -> {out!r}", flush=True)
    except Exception as exc:  # noqa: BLE001
        results["image"] = False
        print(f"[modality] image FAILED: {exc}", flush=True)

    try:
        aud = _chat([
            {"type": "text", "text": "Did you receive audio? Answer yes or no."},
            {"type": "audio_url", "audio_url": {"url": _wav_data_uri()}},
        ])
        out = aud["choices"][0]["message"]["content"]
        results["audio"] = bool(out)
        print(f"[modality] audio -> {out!r}", flush=True)
    except Exception as exc:  # noqa: BLE001
        results["audio"] = False
        print(f"[modality] audio FAILED: {exc}", flush=True)

    ok = all(results.get(k) for k in ("models_ready", "text", "image", "audio"))
    print(f"[modality] SUMMARY {json.dumps(results)} ALL_OK={ok}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
