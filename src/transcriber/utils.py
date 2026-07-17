# SPDX-License-Identifier: MIT
"""Shared utility functions used across transcriber modules."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from transcriber.models import _THINK_TAGS_PATTERN, SPACE_PATTERN


def eprint(*args: object) -> None:
    print(*args, file=sys.stderr)


def normalise_whitespace(text: str) -> str:
    return SPACE_PATTERN.sub(" ", text).strip()


def strip_think_tags(text: str) -> str:
    """Strip <think>...</think> blocks that reasoning models may leak."""
    return _THINK_TAGS_PATTERN.sub("", text).strip()


def format_clock(seconds: float, decimal: str = ".") -> str:
    seconds = max(0.0, seconds)
    millis = int(round(seconds * 1000.0))
    hours, remainder = divmod(millis, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, ms = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{decimal}{ms:03d}"


def read_optional_text(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise RuntimeError(f"Could not read {path}: {exc}") from exc


def parse_glossary(path: Path | None, inline_terms: str) -> list[str]:
    terms: list[str] = []
    if path:
        content = read_optional_text(path)
        for line in content.splitlines():
            line = line.strip().lstrip("-•*").strip()
            if line and not line.startswith("#"):
                terms.append(line)
    if inline_terms:
        terms.extend(term.strip() for term in inline_terms.split(",") if term.strip())
    return list(dict.fromkeys(terms))


def run_ffmpeg(source: Path, destination: Path, normalise_audio: bool) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("FFmpeg is not installed. Install it with: sudo apt install ffmpeg")

    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-i",
        str(source),
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
    ]
    if normalise_audio:
        command.extend(["-af", "loudnorm=I=-16:TP=-1.5:LRA=11"])
    command.extend(["-c:a", "pcm_s16le", str(destination)])

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or "unknown FFmpeg error"
        raise RuntimeError(f"FFmpeg could not decode the input: {detail}")
