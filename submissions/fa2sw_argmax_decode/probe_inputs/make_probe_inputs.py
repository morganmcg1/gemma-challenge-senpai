#!/usr/bin/env python3
"""Generate the minimal staged media samples the functional modalities probe uses.

These are deliberately tiny — a fraction-of-a-second mono tone and an 8-frame
32x32 clip — just enough to exercise the audio and video towers of the served
``google/gemma-4-E4B-it`` endpoint without storing large binaries in the repo.
Regenerate with the *server* venv python (it carries ``cv2`` for the MP4 writer;
the WAV uses only the stdlib ``wave`` module):

    /tmp/senpai-venvs/<deps-hash>/bin/python \
        submissions/fa2sw_precache_kenyan/probe_inputs/make_probe_inputs.py

The bytes are committed, so the functional probe never depends on this script at
run time — it only reads ``probe_audio.wav`` / ``probe_video.mp4``.
"""
from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Audio: 0.25 s, 16 kHz mono 16-bit, a low-amplitude 440 Hz tone (real signal so
# the audio encoder has structure to attend to, not flat silence).
AUDIO_PATH = HERE / "probe_audio.wav"
AUDIO_RATE = 16000
AUDIO_SECONDS = 0.25
AUDIO_FREQ = 440.0
AUDIO_AMP = 0.3

# Video: 8 frames, 32x32, ~8 fps — a diagonal gradient that shifts each frame so
# there is temporal structure for the (shared vision) video pathway.
VIDEO_PATH = HERE / "probe_video.mp4"
VIDEO_SIZE = 32
VIDEO_FRAMES = 8
VIDEO_FPS = 8


def write_audio() -> None:
    n = int(AUDIO_RATE * AUDIO_SECONDS)
    frames = bytearray()
    for i in range(n):
        sample = int(AUDIO_AMP * 32767 * math.sin(2 * math.pi * AUDIO_FREQ * i / AUDIO_RATE))
        frames += struct.pack("<h", sample)
    with wave.open(str(AUDIO_PATH), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(AUDIO_RATE)
        w.writeframes(bytes(frames))
    print(f"wrote {AUDIO_PATH} ({AUDIO_PATH.stat().st_size} bytes)")


def write_video() -> None:
    import cv2  # noqa: PLC0415 — authoring-only dependency (server venv)
    import numpy as np  # noqa: PLC0415

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(VIDEO_PATH), fourcc, VIDEO_FPS, (VIDEO_SIZE, VIDEO_SIZE))
    if not writer.isOpened():
        raise RuntimeError("cv2.VideoWriter failed to open (mp4v codec unavailable)")
    for f in range(VIDEO_FRAMES):
        frame = np.zeros((VIDEO_SIZE, VIDEO_SIZE, 3), dtype=np.uint8)
        for y in range(VIDEO_SIZE):
            for x in range(VIDEO_SIZE):
                frame[y, x] = ((x * 8 + f * 16) % 256, (y * 8) % 256, 128)
        writer.write(frame)
    writer.release()
    print(f"wrote {VIDEO_PATH} ({VIDEO_PATH.stat().st_size} bytes)")


if __name__ == "__main__":
    write_audio()
    write_video()
