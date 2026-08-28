"""Filesystem layout for the Umbrel deployment.

Everything the app writes lives under DATA_DIR, which Umbrel binds to
${APP_DATA_DIR} on the host, so model weights, caches and job state all
survive app updates and restarts.
"""

from __future__ import annotations

import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("DATA_DIR", "/app/data"))

JOBS_DIR = DATA_DIR / "jobs"
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", str(DATA_DIR / "output")))
CACHE_DIR = DATA_DIR / "cache"
MODELS_DIR = DATA_DIR / "models"
DB_PATH = DATA_DIR / "cleancut.db"

# Host media, bind-mounted read/write by docker-compose. Colon-separated so a
# user can add their own roots without touching the code.
_DEFAULT_ROOTS = "/media/network:/media/home"


def media_roots() -> list[Path]:
    """Existing media roots, in display order."""
    raw = os.environ.get("MEDIA_ROOTS", _DEFAULT_ROOTS)
    roots: list[Path] = []
    for part in raw.split(":"):
        part = part.strip()
        if not part:
            continue
        p = Path(part)
        if p.is_dir():
            roots.append(p)
    return roots


def ensure_dirs() -> None:
    for d in (DATA_DIR, JOBS_DIR, OUTPUT_DIR, CACHE_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def job_dir(job_id: int) -> Path:
    return JOBS_DIR / str(job_id)
