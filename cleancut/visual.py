"""Visual scene detection using NudeNet on sampled frames.

Two scanning modes:
1. Streak mode (no shot list): per-second sampling, emit a cut when at least
   `min_streak` consecutive samples all hit. Robust against single-frame
   false positives.
2. Shot-aware mode: given a shot list from PySceneDetect, evaluate each shot
   independently. A shot is cut when at least `shot_hit_fraction` of its
   sampled frames are flagged. Far more robust because we never cut mid-shot.
"""

from __future__ import annotations

from pathlib import Path

from tqdm import tqdm

from cleancut.config import Config
from cleancut.edl import EditDecision, EditDecisionList
from cleancut.scenes import Shot

# NudeNet class names that signal explicit content. Trip a cut on hit.
EXPLICIT_CLASSES = {
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "BUTTOCKS_EXPOSED",
    "ANUS_EXPOSED",
}


def _open_video_and_detector(video_path: Path):
    try:
        import cv2  # type: ignore
        from nudenet import NudeDetector  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "Visual detection requires extras. Install with: pip install -e '.[visual]'"
        ) from e

    detector = NudeDetector()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    return cv2, detector, cap


def _is_explicit(detections, threshold: float) -> bool:
    return any(
        d.get("class") in EXPLICIT_CLASSES and float(d.get("score", 0)) >= threshold
        for d in detections
    )


def _detect_on_frame(detector, cv2, frame) -> list[dict]:
    """NudeNet's detect() signature varies by build — some want a path."""
    try:
        return detector.detect(frame)
    except Exception:
        import os
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            try:
                cv2.imwrite(tmp.name, frame)
                return detector.detect(tmp.name)
            finally:
                os.unlink(tmp.name)


def scan_video(
    video_path: Path,
    config: Config,
    shots: list[Shot] | None = None,
    use_cache: bool = True,
) -> EditDecisionList:
    """Sample frames and produce cut decisions. Shot-aware when shots given.

    Caches the full result EDL keyed by (video, threshold, sample rate, mode,
    streak/fraction, shot count + first/last shot times). Re-runs with the
    same params skip the 1600+ inferences entirely.
    """
    from cleancut import cache as _cache
    from dataclasses import asdict

    mode = "shot-aware" if shots else "streak"
    shot_fingerprint = None
    if shots:
        shot_fingerprint = {
            "n": len(shots),
            "first": (shots[0].start, shots[0].end),
            "last": (shots[-1].start, shots[-1].end),
        }
    h = _cache.config_hash(
        mode=mode,
        threshold=config.visual_threshold,
        sample_seconds=config.visual_sample_seconds,
        min_streak=config.visual_min_streak,
        hit_fraction=config.visual_shot_hit_fraction,
        action=config.actions.get("nudity", "cut"),
        shots=shot_fingerprint,
    )
    if use_cache:
        hit = _cache.load(video_path, "nudenet", h)
        if hit:
            return EditDecisionList(
                decisions=[EditDecision(**d) for d in hit.get("decisions", [])]
            )

    cv2, detector, cap = _open_video_and_detector(video_path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = total_frames / fps if fps > 0 else 0.0
        action = config.actions.get("nudity", "cut")

        if shots:
            edl = _shot_aware_scan(
                cv2, detector, cap, fps, shots, config, action
            )
        else:
            edl = _streak_scan(
                cv2, detector, cap, fps, duration, config, action
            )
    finally:
        cap.release()

    if use_cache:
        _cache.save(video_path, "nudenet", h, {
            "decisions": [asdict(d) for d in edl.decisions],
        })
    return edl


def _iter_sampled_frames(cap, fps, samples):
    """Decode sequentially, yielding (payload, frame) for each sample.

    `samples` is an ascending list of (time_seconds, payload) pairs. Skipping
    with grab() is far cheaper than CAP_PROP_POS_FRAMES seeks, which force a
    keyframe seek + decode-forward for every sample (~14k per movie at the
    thorough preset).
    """
    pos = 0
    for t, payload in samples:
        target = int(round(t * fps))
        while pos < target:
            if not cap.grab():
                return
            pos += 1
        ok, frame = cap.read()
        pos += 1
        if not ok or frame is None:
            continue
        yield payload, frame


def _streak_scan(cv2, detector, cap, fps, duration, config: Config, action: str) -> EditDecisionList:
    """No shot info: walk the video by `visual_sample_seconds`; require
    `visual_min_streak` consecutive hits before emitting a cut."""
    sample_step = config.visual_sample_seconds
    min_streak = max(1, config.visual_min_streak)
    edl = EditDecisionList()

    n_samples = int(duration / sample_step) if duration > 0 else 0
    samples = [(i * sample_step, i * sample_step) for i in range(n_samples)]
    streak_start: float | None = None
    streak_classes: set[str] = set()
    last_hit_t: float | None = None

    for t, frame in _iter_sampled_frames(
        cap, fps, tqdm(samples, desc="Visual scan", unit="frame", leave=False)
    ):
        detections = _detect_on_frame(detector, cv2, frame)
        hits = [
            d for d in detections
            if d.get("class") in EXPLICIT_CLASSES
            and float(d.get("score", 0)) >= config.visual_threshold
        ]
        if hits:
            if streak_start is None:
                streak_start = t
            streak_classes.update(d["class"] for d in hits)
            last_hit_t = t
        else:
            if streak_start is not None and last_hit_t is not None:
                _emit_if_long_enough(
                    edl, streak_start, last_hit_t + sample_step,
                    streak_classes, min_streak, sample_step, action,
                )
                streak_start = None
                streak_classes = set()

    if streak_start is not None and last_hit_t is not None:
        _emit_if_long_enough(
            edl, streak_start, last_hit_t + sample_step,
            streak_classes, min_streak, sample_step, action,
        )

    return edl


def _emit_if_long_enough(
    edl: EditDecisionList,
    start: float,
    end: float,
    classes: set[str],
    min_streak: int,
    sample_step: float,
    action: str,
) -> None:
    n_samples = max(1, round((end - start) / sample_step))
    if n_samples >= min_streak:
        edl.add(
            EditDecision(
                start=start,
                end=end,
                action=action,
                category="nudity",
                reason=f"visual streak ({n_samples} frames): {', '.join(sorted(classes))}",
                source="visual",
            )
        )


def _shot_aware_scan(
    cv2, detector, cap, fps, shots: list[Shot], config: Config, action: str,
) -> EditDecisionList:
    """For each shot: sample frames, compute hit fraction, cut shot if over threshold.

    All samples are decoded in one sequential pass over the video rather than
    per-shot random seeks.
    """
    edl = EditDecisionList()
    sample_step = config.visual_sample_seconds
    min_fraction = config.visual_shot_hit_fraction
    min_frames_in_shot = 2  # even tiny shots get two samples

    # Plan every sample up front: (time, shot index), ascending.
    samples: list[tuple[float, int]] = []
    n_planned: dict[int, int] = {}
    for idx, shot in enumerate(shots):
        if shot.duration <= 0:
            continue
        n = max(min_frames_in_shot, int(shot.duration / sample_step))
        n_planned[idx] = n
        samples += [(shot.start + (shot.duration * (k + 0.5) / n), idx) for k in range(n)]
    samples.sort(key=lambda x: x[0])

    n_hits: dict[int, int] = {}
    hit_classes: dict[int, set[str]] = {}
    for idx, frame in _iter_sampled_frames(
        cap, fps, tqdm(samples, desc="Visual scan (shots)", unit="frame", leave=False)
    ):
        detections = _detect_on_frame(detector, cv2, frame)
        hits = [
            d for d in detections
            if d.get("class") in EXPLICIT_CLASSES
            and float(d.get("score", 0)) >= config.visual_threshold
        ]
        if hits:
            n_hits[idx] = n_hits.get(idx, 0) + 1
            hit_classes.setdefault(idx, set()).update(d["class"] for d in hits)

    for idx, shot in enumerate(shots):
        hits_in_shot = n_hits.get(idx, 0)
        planned = n_planned.get(idx, 0)
        fraction = hits_in_shot / max(1, planned)
        # The >= 2 floor means a two-sample shot needs both frames flagged —
        # single-frame hits never cut a shot.
        if fraction >= min_fraction and hits_in_shot >= 2:
            edl.add(
                EditDecision(
                    start=shot.start,
                    end=shot.end,
                    action=action,
                    category="nudity",
                    reason=f"shot {hits_in_shot}/{planned} frames flagged: "
                           f"{', '.join(sorted(hit_classes.get(idx, set())))}",
                    source="visual-shot",
                )
            )

    return edl
