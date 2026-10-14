"""Browsing the mounted media roots.

Every path that arrives from the browser is resolved and checked against the
configured roots before it is opened -- the app runs behind Umbrel's proxy on
a trusted LAN, but a traversal bug here would hand out the whole container
filesystem, so containment is enforced in one place.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from webapp.paths import OUTPUT_DIR, media_roots

VIDEO_SUFFIXES = {
    ".mp4", ".mkv", ".m4v", ".mov", ".avi", ".webm", ".mpg", ".mpeg", ".ts", ".wmv", ".flv",
}
MAX_BATCH_FILES = 500


def folder_videos(path: Path, recursive: bool = True) -> list[dict[str, Any]]:
    """Expand a folder without following directory links or NAS housekeeping trees.

    Fail rather than silently submit a truncated or partly unreadable folder.
    Every discovered file is independently checked against the media roots.
    """
    found: dict[str, dict[str, Any]] = {}

    def failed(error):
        raise ValueError("Cannot read the whole folder. Check NAS permissions and try again.") from error

    for count, (directory, dirs, names) in enumerate(os.walk(path, onerror=failed), 1):
        if count > 10_000:
            raise ValueError("Folder is too large to browse. Choose a smaller subfolder.")
        dirs[:] = sorted(d for d in dirs if not d.startswith((".", "@"))
                         and not (Path(directory) / d).is_symlink()) if recursive else []
        for name in sorted(names, key=str.casefold):
            candidate = Path(directory) / name
            if name.startswith((".", "@")) or not is_video(candidate) or candidate.stem.lower().endswith(".clean"):
                continue
            video = resolve_safe(str(candidate))
            if video is None or not video.is_file():
                continue
            if not os.access(video, os.R_OK):
                raise ValueError(f"Video is not readable: {name}. Check NAS permissions.")
            found[str(video)] = _entry(video, "file")
            if len(found) > MAX_BATCH_FILES:
                raise ValueError(f"Select at most {MAX_BATCH_FILES} videos at once. Choose a smaller folder.")
    return list(found.values())


def allowed_roots() -> list[Path]:
    """Media roots plus the app's own output directory, which is browsable so a
    cleaned file can be re-scanned or picked up again after a restart."""
    roots = list(media_roots())
    out = OUTPUT_DIR
    if out.is_dir() and not any(_within(out, r) for r in roots):
        roots.append(out)
    return roots


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def resolve_safe(raw: str) -> Path | None:
    """Resolve a client-supplied path, or None if it escapes every root."""
    if not raw:
        return None
    try:
        candidate = Path(raw).resolve()
    except (OSError, RuntimeError):
        return None
    for root in allowed_roots():
        try:
            root_resolved = root.resolve()
        except OSError:
            continue
        if candidate == root_resolved or _within(candidate, root_resolved):
            return candidate
    return None


def is_video(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_SUFFIXES


def _entry(path: Path, kind: str) -> dict[str, Any]:
    info: dict[str, Any] = {"name": path.name, "path": str(path), "kind": kind}
    if kind == "file":
        try:
            info["size"] = path.stat().st_size
        except OSError:
            info["size"] = 0
    return info


def list_roots() -> list[dict[str, Any]]:
    return [_entry(r, "dir") for r in allowed_roots()]


def list_dir(path: Path) -> dict[str, Any]:
    """One directory listing: subdirectories and video files, name-sorted."""
    dirs: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []
    try:
        for child in sorted(path.iterdir(), key=lambda p: p.name.lower()):
            if child.name.startswith(".") or child.name.startswith("@"):
                continue
            try:
                if child.is_dir():
                    dirs.append(_entry(child, "dir"))
                elif child.is_file() and is_video(child):
                    files.append(_entry(child, "file"))
            except OSError:
                continue
    except (PermissionError, OSError) as e:
        return {"path": str(path), "error": str(e), "dirs": [], "files": [], "parent": None}

    parent = str(path.parent) if resolve_safe(str(path.parent)) else None
    return {"path": str(path), "parent": parent, "dirs": dirs, "files": files, "error": None}


def search(term: str, limit: int = 200, max_dirs: int = 20_000) -> list[dict[str, Any]]:
    """Recursive filename search across the roots.

    os.walk with in-place pruning rather than rglob: the media root is usually a
    network share holding a NAS's snapshot and recycle directories, and rglob
    descends into those in full before any name filter gets to reject them. The
    directory budget bounds a search that would otherwise crawl a mounted share
    on every keystroke.
    """
    term = term.strip().lower()
    if len(term) < 2:
        return []
    hits: list[dict[str, Any]] = []
    scanned = 0
    for root in allowed_roots():
        for dirpath, dirnames, filenames in os.walk(root, onerror=lambda e: None):
            scanned += 1
            if len(hits) >= limit or scanned > max_dirs:
                return hits
            # Prune in place so os.walk never descends into them at all.
            dirnames[:] = [d for d in dirnames if not d.startswith(".") and not d.startswith("@")]
            for name in filenames:
                if name.startswith(".") or term not in name.lower():
                    continue
                path = Path(dirpath) / name
                if is_video(path):
                    hits.append(_entry(path, "file"))
                    if len(hits) >= limit:
                        return hits
    return hits
