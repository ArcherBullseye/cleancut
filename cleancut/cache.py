"""Disk cache for expensive intermediate results.

Caches live in `~/.cache/cleancut/` keyed by a hash of the source video's
absolute path. Each entry stores the video's mtime + size + a per-feature
config hash. A cache hit requires all three to match the live file/config —
any change invalidates.

Used by:
- PySceneDetect shot boundaries (`shots`)
- NudeNet visual scan results (`nudenet`)
- Whisper transcription (`whisper`)
- LLM dialogue classification (`llm_dialogue`)
- VLM scene classification (`vlm`)
- AST audio events (`audio_events`)

`save` and `load` work on plain Python dicts; the caller is responsible for
serializing dataclasses to/from JSON-friendly forms.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


CACHE_DIR = Path(os.environ.get("CLEANCUT_CACHE_DIR", str(Path.home() / ".cache" / "cleancut")))


def _video_fingerprint(video: Path) -> dict[str, Any]:
    st = video.stat()
    return {"path": str(video.resolve()), "size": st.st_size, "mtime": int(st.st_mtime)}


def _cache_key(video: Path, feature: str, config_hash: str) -> Path:
    # The config hash is part of the filename so entries for different
    # configs coexist — toggling between two configs must not thrash.
    h = hashlib.sha256(str(video.resolve()).encode("utf-8")).hexdigest()[:16]
    return CACHE_DIR / f"{h}.{feature}.{config_hash}.json"


def config_hash(**fields: Any) -> str:
    """Stable short hash of a config dict — pass only the fields that affect output."""
    payload = json.dumps(fields, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def load(video: Path, feature: str, expected_config_hash: str) -> dict | None:
    """Return cached payload dict, or None on miss / stale / invalidated."""
    path = _cache_key(video, feature, expected_config_hash)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if data.get("config_hash") != expected_config_hash:
        return None
    fp = _video_fingerprint(video)
    if data.get("video") != fp:
        return None
    return data.get("payload")


def save(video: Path, feature: str, config_hash_val: str, payload: dict) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_key(video, feature, config_hash_val)
    # Write-temp-then-rename so an interrupted save can't leave corrupt JSON.
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({
        "video": _video_fingerprint(video),
        "config_hash": config_hash_val,
        "payload": payload,
    }, indent=2))
    tmp.replace(path)
    return path


def clear(video: Path | None = None, feature: str | None = None) -> int:
    """Remove cache files; returns count removed.

    With no args: wipe everything. With `video`: only entries for that video.
    With both: only the specific feature for that video.
    """
    if not CACHE_DIR.exists():
        return 0
    n = 0
    if video is None:
        for p in CACHE_DIR.glob("*.json"):
            p.unlink()
            n += 1
        return n
    h = hashlib.sha256(str(video.resolve()).encode("utf-8")).hexdigest()[:16]
    pattern = f"{h}.{feature}.*.json" if feature else f"{h}.*.json"
    for p in CACHE_DIR.glob(pattern):
        p.unlink()
        n += 1
    return n
