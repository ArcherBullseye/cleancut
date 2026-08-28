"""User-editable settings, persisted as JSON in DATA_DIR.

Kept separate from the job database so a corrupt settings file can never take
the job queue down with it.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any

from webapp.paths import DATA_DIR

_SETTINGS_PATH = DATA_DIR / "settings.json"
_lock = threading.Lock()

# Umbrel puts every app container on the same bridge network, so the Ollama
# community app is reachable by container name. Empty string disables the
# LLM/VLM signals entirely.
DEFAULT_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://ollama_ollama_1:11434")

DEFAULTS: dict[str, Any] = {
    "preset": "balanced",
    "ollama_host": DEFAULT_OLLAMA_HOST,
    "llm_model": "llama3.1:8b",
    "vlm_model": "llava:7b",
    # Per-category action. "keep" means detected but never edited.
    "actions": {
        "profanity": "mute",
        "drugs": "mute",
        "sex": "mute",
        "violence": "keep",
        "nudity": "cut",
    },
    "categories": ["profanity", "drugs", "sex", "nudity"],
    # libx264 is the only sane choice on the Umbrel Home -- no GPU, and
    # videotoolbox is macOS-only.
    "encoder": "libx264",
    "quality": 20,
    "prefer_language": "eng",
    # How the softened subtitles reach the output.
    #   soft -- a toggleable track in the container (default)
    #   burn -- painted into the picture, irreversible
    #   none -- no subtitles at all
    "subtitle_mode": "soft",
    "output_dir": "",
    "auto_render": False,
}


def _read() -> dict[str, Any]:
    if not _SETTINGS_PATH.exists():
        return {}
    try:
        data = json.loads(_SETTINGS_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def load() -> dict[str, Any]:
    """Stored settings merged over the defaults."""
    merged = json.loads(json.dumps(DEFAULTS))
    stored = _read()
    for key, value in stored.items():
        if key not in DEFAULTS:
            continue
        if isinstance(DEFAULTS[key], dict) and isinstance(value, dict):
            merged[key].update(value)
        else:
            merged[key] = value
    return merged


def save(updates: dict[str, Any]) -> dict[str, Any]:
    with _lock:
        current = load()
        for key, value in updates.items():
            if key not in DEFAULTS:
                continue
            if isinstance(DEFAULTS[key], dict) and isinstance(value, dict):
                current[key].update(value)
            else:
                current[key] = value
        _SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = _SETTINGS_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(current, indent=2))
        tmp.replace(_SETTINGS_PATH)
    return current
