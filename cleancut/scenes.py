"""Shot boundary detection using PySceneDetect.

Used for two things:
1. Snapping dialogue-based cuts outward to the nearest shot boundary, so cuts
   land on natural edits rather than mid-shot.
2. Snapping temporally confirmed NudeNet ranges to complete shots without using
   whole-shot hit percentages that can hide brief nudity.
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


def detect_shots(
    video_path: Path, threshold: float = 27.0, use_cache: bool = True, *, include_single_shot: bool = False,
) -> list[Shot]:
    """Return all detected shot boundaries.

    `threshold` is the PySceneDetect ContentDetector threshold. Lower = more
    sensitive (more cuts). 27 is the library default; 30 is more conservative.

    Results are cached under ~/.cache/cleancut/ keyed by video path/mtime/size
    and the threshold. Cache survives across runs and across cleancut versions
    as long as the params match.
    """
    from cleancut import cache as _cache

    h = _cache.config_hash(threshold=threshold, kind="ContentDetector",
                           **({"include_single_shot": True} if include_single_shot else {}))
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

    scene_list = detect(str(video_path), ContentDetector(threshold=threshold),
                        start_in_scene=include_single_shot)
    if include_single_shot and scene_list:
        shots = _source_timed_shots(video_path, scene_list)
    else:
        shots = [Shot(start=float(s.get_seconds()), end=float(e.get_seconds())) for s, e in scene_list]
    # Empty results cache too — otherwise a shotless video re-runs the full
    # pass on every scan.
    if use_cache:
        _cache.save(video_path, "shots", h, {
            "shots": [{"start": s.start, "end": s.end} for s in shots],
            "count": len(shots),
        })
    return shots


def _source_timed_shots(video: Path, scenes) -> list[Shot]:
    import cv2

    from cleancut.probe import probe_video_timing
    from cleancut.visual import _open_capture, iter_frame_positions

    offset, duration, video_end = probe_video_timing(video)
    bounds = {time.get_frames() for scene in scenes for time in scene}
    times = {}
    cap = _open_capture(cv2, video, source_timestamps=True)
    try:
        for index, timestamp in iter_frame_positions(cv2, cap, offset):
            if index in bounds:
                times[index] = max(0.0, timestamp)
        if index + 1 in bounds:
            times[index + 1] = min(duration, video_end) if video_end is not None else duration
    finally:
        cap.release()
    try:
        return [Shot(times[s.get_frames()], times[e.get_frames()]) for s, e in scenes]
    except KeyError as exc:
        raise RuntimeError("Could not align shot boundaries with the source frames.") from exc


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
