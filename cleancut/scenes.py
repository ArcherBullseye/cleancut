"""Shot boundary detection using PySceneDetect.

Used for two things:
1. Snapping dialogue-based cuts outward to the nearest shot boundary, so cuts
   land on natural edits rather than mid-shot.
2. Aggregating visual NudeNet hits per shot — a shot is cut only when a
   significant fraction of its sampled frames are flagged.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Shot:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def contains(self, t: float) -> bool:
        return self.start <= t < self.end


def detect_shots(video_path: Path, threshold: float = 27.0, use_cache: bool = True) -> list[Shot]:
    """Return all detected shot boundaries.

    `threshold` is the PySceneDetect ContentDetector threshold. Lower = more
    sensitive (more cuts). 27 is the library default; 30 is more conservative.

    Results are cached under ~/.cache/cleancut/ keyed by video path/mtime/size
    and the threshold. Cache survives across runs and across cleancut versions
    as long as the params match.
    """
    from cleancut import cache as _cache

    h = _cache.config_hash(threshold=threshold, kind="ContentDetector")
    if use_cache:
        hit = _cache.load(video_path, "shots", h)
        if hit:
            return [Shot(start=s["start"], end=s["end"]) for s in hit.get("shots", [])]

    try:
        from scenedetect import ContentDetector, detect  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "Scene detection requires extras. Install with: pip install -e '.[scenes]'"
        ) from e

    scene_list = detect(str(video_path), ContentDetector(threshold=threshold))
    shots = [Shot(start=float(s.get_seconds()), end=float(e.get_seconds())) for s, e in scene_list]
    # Empty results cache too — otherwise a shotless video re-runs the full
    # pass on every scan.
    if use_cache:
        _cache.save(video_path, "shots", h, {
            "shots": [{"start": s.start, "end": s.end} for s in shots],
            "count": len(shots),
        })
    return shots


def shot_containing(t: float, shots: list[Shot]) -> Shot | None:
    """Return the shot containing `t`, or None if outside all shots."""
    # Binary search would be faster, but shot lists are small (~1-5k for a movie).
    for s in shots:
        if s.contains(t):
            return s
    return None


def snap_range_to_shots(start: float, end: float, shots: list[Shot]) -> tuple[float, float]:
    """Extend [start, end] outward to the nearest shot boundaries that fully
    contain the range. If shots is empty, return the range unchanged."""
    if not shots:
        return start, end
    s_shot = shot_containing(start, shots)
    e_shot = shot_containing(max(start, end - 1e-6), shots)
    new_start = s_shot.start if s_shot else start
    new_end = e_shot.end if e_shot else end
    return new_start, new_end
