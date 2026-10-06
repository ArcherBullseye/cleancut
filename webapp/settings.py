"""User-editable settings, persisted as JSON in DATA_DIR.

Kept separate from the job database so a corrupt settings file can never take
the job queue down with it.
"""

from __future__ import annotations

import json
import os
import platform
import threading
from typing import Any

from webapp.paths import DATA_DIR

_SETTINGS_PATH = DATA_DIR / "settings.json"
_lock = threading.Lock()

_IS_MAC = platform.system() == "Darwin"

# Native macOS talks to the local Ollama service. Umbrel reaches its community
# app over the Docker bridge. Empty string disables both AI passes.
DEFAULT_OLLAMA_HOST = os.environ.get(
    "OLLAMA_HOST",
    "http://127.0.0.1:11434" if _IS_MAC else "http://ollama_ollama_1:11434",
)

DEFAULTS: dict[str, Any] = {
    "preset": "balanced",
    "ollama_host": DEFAULT_OLLAMA_HOST,
    # Qwen 3.5 is multimodal, so the Mac can reuse one resident local model for
    # dialogue and image classification. Keep lightweight Umbrel defaults.
    "llm_model": "qwen3.5:9b" if _IS_MAC else "llama3.2:3b",
    "vlm_model": "qwen3.5:9b" if _IS_MAC else "moondream",
    "local_ai_enabled": _IS_MAC,
    # Per-category action. "keep" means detected but never edited.
    "actions": {
        "profanity": "mute",
        "drugs": "mute",
        "sex": "mute",
        "violence": "keep",
        "nudity": "cut",
    },
    "categories": ["profanity", "drugs", "sex", "nudity"],
    # Resolves to VideoToolbox on native macOS and libx264 in Umbrel/Linux.
    # HDR sources automatically select 10-bit HEVC VideoToolbox on macOS.
    "encoder": "auto",
    "quality": 20,
    "analysis_height": 720,
    "analysis_proxy": True,
    "nudity_model": "accurate",
    "render_validation": "full" if platform.system() == "Darwin" else "quick",
    "prefer_language": "eng",
    # How the softened subtitles reach the output.
    #   soft -- a toggleable track in the container (default)
    #   burn -- painted into the picture, irreversible
    #   none -- no subtitles at all
    "subtitle_mode": "soft",
    # Native Mac defaults to returning the cleaned copy beside the mounted NAS
    # source. Umbrel keeps its managed output directory default.
    "output_location": "source" if _IS_MAC else "folder",
    "output_dir": "",
    "auto_render": False,
    "profanity_audio": "mute",
    "speech_host": os.environ.get("CLEANCUT_SPEECH_HOST", "http://127.0.0.1:8765"),
    "speech_model": "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit",
    "speech_token": os.environ.get("CLEANCUT_SPEECH_TOKEN", ""),
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
    # Migrate only the exact legacy Mac defaults. User-selected model pairs and
    # custom hosts remain untouched.
    if _IS_MAC and stored.get("ollama_host") == "http://ollama_ollama_1:11434":
        stored["ollama_host"] = "http://127.0.0.1:11434"
    if _IS_MAC and (
        stored.get("llm_model"), stored.get("vlm_model")
    ) == ("llama3.2:3b", "moondream"):
        stored["llm_model"] = "qwen3.5:9b"
        stored["vlm_model"] = "qwen3.5:9b"
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
            if key == "encoder" and value not in {
                "auto", "videotoolbox", "hevc_videotoolbox", "libx264", "libx265"
            }:
                continue
            if key == "render_validation" and value not in {"none", "quick", "full"}:
                continue
            if key == "nudity_model" and value not in {"accurate", "fast"}:
                continue
            if key == "output_location" and value not in {"source", "folder"}:
                continue
            if key == "profanity_audio" and value not in {"mute", "replace"}:
                continue
            if key in {"speech_host", "speech_model", "speech_token"}:
                if not isinstance(value, str) or len(value) > 512:
                    continue
                value = value.strip()
            if key == "analysis_height":
                try:
                    value = max(360, min(1080, int(value)))
                except (TypeError, ValueError):
                    continue
            if isinstance(DEFAULTS[key], dict) and isinstance(value, dict):
                current[key].update(value)
            else:
                current[key] = value
        _SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = _SETTINGS_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(current, indent=2))
        # Settings may contain the shared speech credential.
        tmp.chmod(0o600)
        tmp.replace(_SETTINGS_PATH)
    return current
