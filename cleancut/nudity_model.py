"""Local model management for NudeNet.

The more accurate 640m model is an official NudeNet release asset.  It is
downloaded once, checksum-verified, and then used entirely offline.  The small
model bundled with the ``nudenet`` package is always available as a fallback.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

NUDENET_640M_URL = (
    "https://api.github.com/repos/notAI-tech/NudeNet/releases/assets/176832019"
)
NUDENET_640M_SHA256 = (
    "04fe3d77980780c1f8297dc6d7f942fd5b3abe6942a188f742a85241e4f634eb"
)
NUDENET_640M_SIZE = 103_538_690
NUDENET_640M_FILENAME = "nudenet-640m.onnx"


@dataclass(frozen=True)
class NudityModel:
    name: str
    path: Path
    resolution: int


def model_dir() -> Path:
    explicit = os.environ.get("CLEANCUT_MODEL_DIR")
    if explicit:
        return Path(explicit).expanduser()
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return cache / "cleancut" / "models"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_accurate_model(path: Path) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == NUDENET_640M_SIZE
        and _sha256(path) == NUDENET_640M_SHA256
    )


def ensure_accurate_model() -> Path:
    """Return the verified 640m model, downloading it atomically if needed."""
    destination = model_dir() / NUDENET_640M_FILENAME
    if _valid_accurate_model(destination):
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(".onnx.partial")
    try:
        partial.unlink(missing_ok=True)
        request = urllib.request.Request(
            NUDENET_640M_URL,
            headers={
                "Accept": "application/octet-stream",
                "User-Agent": "CleanCut-local-nudity-scanner",
            },
        )
        digest = hashlib.sha256()
        size = 0
        print("Downloading the free NudeNet 640m model (99 MB, one time)...")
        with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as out:
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                out.write(chunk)
                digest.update(chunk)
                size += len(chunk)
        if size != NUDENET_640M_SIZE or digest.hexdigest() != NUDENET_640M_SHA256:
            raise RuntimeError("downloaded NudeNet model failed size/checksum validation")
        partial.replace(destination)
        return destination
    finally:
        partial.unlink(missing_ok=True)


def bundled_model() -> NudityModel:
    try:
        path = Path(resources.files("nudenet").joinpath("320n.onnx"))
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "Visual detection requires extras. Install with: pip install -e '.[visual]'"
        ) from exc
    return NudityModel("NudeNet-320n", path, 320)


def resolve_nudity_model(requested: str) -> NudityModel:
    """Resolve an entirely local model, falling back safely when offline."""
    if requested == "fast":
        return bundled_model()
    try:
        return NudityModel("NudeNet-640m", ensure_accurate_model(), 640)
    except Exception as exc:  # noqa: BLE001 - network/filesystem failures all fall back locally
        print(f"Warning: accurate NudeNet model unavailable ({exc}); using bundled 320n model.")
        return bundled_model()
