"""Preserve generated outputs and the input contract for quality comparisons."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import TypedDict


class GeneratedText(TypedDict):
    prompt_idx: int
    iteration: str
    generated_text: str
    truncated: bool


_GENERATED = re.compile(r"^\s*(?:\[\s*INFO\s*\]\s*)?\[(warm-up|\d+)\]\[P(\d+)\] Generated:\s?(.*)$", re.I)
_SECTION = re.compile(r"^\s*\[(?:\s*(?:INFO|WARNING|ERROR|DEBUG)\s*|CMD|monitor|warm-up|\d+)\]")
_MAX_GENERATED_RECORDS = 256
_MAX_GENERATED_TEXT_CHARS = 100_000
_MAX_GENERATED_TOTAL_CHARS = 2_000_000


def generated_texts(output: str) -> list[GeneratedText]:
    """Capture bounded multiline outputs, including a final record without a footer."""
    records: list[GeneratedText] = []
    active: GeneratedText | None = None
    lines: list[str] = []
    total_chars = 0

    def finish_active() -> None:
        nonlocal active, lines, total_chars
        if active is None or len(records) >= _MAX_GENERATED_RECORDS:
            active, lines = None, []
            return
        text = "\n".join(lines).rstrip()
        remaining = max(0, _MAX_GENERATED_TOTAL_CHARS - total_chars)
        limit = min(len(text), _MAX_GENERATED_TEXT_CHARS, remaining)
        active["generated_text"] = text[:limit]
        active["truncated"] = limit < len(text)
        total_chars += limit
        records.append(active)
        active, lines = None, []

    for line in output.splitlines():
        match = _GENERATED.match(line)
        if active is not None and (match or _SECTION.match(line)):
            finish_active()
        if match and len(records) < _MAX_GENERATED_RECORDS:
            active = {
                "prompt_idx": int(match[2]),
                "iteration": match[1].lower(),
                "generated_text": "",
                "truncated": False,
            }
            lines = [match[3]]
        elif active is not None:
            lines.append(line)
    finish_active()
    return records


def generation_fingerprint(
    prompt_path: str, script_path: str, settings: dict[str, object], config_path: str | None = None,
) -> str | None:
    """Snapshot prompt/config/default implementation; never infer old settings."""
    try:
        prompt_bytes = Path(prompt_path).read_bytes()
        prompts = [json.loads(line) for line in prompt_bytes.decode("utf-8").splitlines() if line.strip()]
        if any(any(key in prompt for key in ("image", "images", "video", "audio", "media")) for prompt in prompts):
            return None
        payload = {
            "prompts": hashlib.sha256(prompt_bytes).hexdigest(),
            "script": hashlib.sha256(Path(script_path).read_bytes()).hexdigest(),
            "config": hashlib.sha256(Path(config_path).read_bytes()).hexdigest() if config_path else None,
            "settings": settings,
        }
    except (OSError, ValueError, UnicodeError):
        return None
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()