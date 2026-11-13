"""Local nudity detection using NudeNet.

Scan every source frame or sample the video and rescan around possible hits.
Automatic cuts need repeated confident hits of the same class or one very
strong hit. Repeated weak predictions stay unselected for review.
"""

from __future__ import annotations

import math
import platform
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from itertools import islice
from pathlib import Path

from tqdm import tqdm

from cleancut.config import Config
from cleancut.edl import EditDecision, EditDecisionList
from cleancut.nudity_model import NudityModel, resolve_nudity_model
from cleancut.probe import probe_video_timing
from cleancut.scenes import Shot

EXPLICIT_CLASSES = {
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "BUTTOCKS_EXPOSED",
    "ANUS_EXPOSED",
}


@dataclass(frozen=True)
class NudityEvidence:
    frame: int
    timestamp: float
    box: list[int] | None


def _explicit_hits(detections, threshold: float) -> list[dict]:
    hits = []
    for d in detections:
        try:
            score = float(d.get("score", 0))
            if d.get("class") in EXPLICIT_CLASSES and math.isfinite(score) and threshold <= score <= 1:
                hits.append(d)
        except (TypeError, ValueError, AttributeError):
            continue
    return hits


def _is_explicit(detections, threshold: float) -> bool:
    return bool(_explicit_hits(detections, threshold))


def _open_detector(config: Config, model: NudityModel):
    try:
        import cv2  # type: ignore
        from nudenet import NudeDetector  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Visual detection requires extras. Install with: pip install -e '.[visual]'"
        ) from exc

    detector = NudeDetector(
        model_path=str(model.path), inference_resolution=model.resolution
    )
    backend = "CPU"

    # NudeNet currently ignores its providers argument. Replace the session
    # explicitly on macOS and retain CPU as a transparent fallback.
    if config.nudity_coreml and platform.system() == "Darwin":
        try:
            import onnxruntime as ort  # type: ignore

            if "CoreMLExecutionProvider" in ort.get_available_providers():
                cache_dir = model.path.parent / "coreml-cache"
                cache_dir.mkdir(parents=True, exist_ok=True)
                detector.onnx_session = ort.InferenceSession(
                    str(model.path),
                    providers=[
                        (
                            "CoreMLExecutionProvider",
                            {
                                "ModelFormat": "MLProgram",
                                "MLComputeUnits": "ALL",
                                "RequireStaticInputShapes": "0",
                                "ModelCacheDirectory": str(cache_dir),
                            },
                        ),
                        "CPUExecutionProvider",
                    ],
                )
                detector.input_name = detector.onnx_session.get_inputs()[0].name
                backend = "CoreML"
        except Exception as exc:  # noqa: BLE001 - any provider failure must retain CPU fallback
            print(f"CoreML unavailable for NudeNet ({exc}); using CPU inference.")

    detector._cleancut_backend = backend
    detector._cleancut_model = model.name
    print(f"Nudity detector: {model.name} on {backend}")
    return cv2, detector


def _open_capture(cv2, video_path: Path, *, source_timestamps: bool = False):
    backend = cv2.CAP_FFMPEG if source_timestamps else cv2.CAP_ANY
    cap = cv2.VideoCapture(str(video_path), backend)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open video: {video_path}")
    return cap


def _detect_on_frame(detector, cv2, frame) -> list[dict]:
    """Support NudeNet builds that accept only an image path."""
    try:
        return detector.detect(frame)
    except Exception:  # noqa: BLE001 - compatibility with multiple NudeNet releases
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            try:
                cv2.imwrite(tmp.name, frame)
                return detector.detect(tmp.name)
            finally:
                os.unlink(tmp.name)


def _iter_sampled_frames(cap, fps: float, samples: Iterable[tuple[float, float]]):
    """Decode ascending timestamps sequentially instead of seeking per frame."""
    pos = 0
    for t, payload in samples:
        target = round(t * fps)
        if target < pos:
            # Two nearby timestamps can address the same decoded frame.
            # Counting the next frame under that timestamp invents support.
            continue
        while pos < target:
            if not cap.grab():
                return
            pos += 1
        ok, frame = cap.read()
        pos += 1
        if ok and frame is not None:
            yield payload, frame


def _infer_samples(
    detector,
    cv2,
    cap,
    fps: float,
    samples: list[tuple[float, float]],
    batch_size: int,
    description: str,
) -> list[tuple[float, list[dict]]]:
    return list(_infer_frames(
        detector, cv2, _iter_sampled_frames(cap, fps, samples), batch_size, description,
        total=len(samples),
    ))


def _infer_frames(detector, cv2, frames, batch_size: int, description: str, *, total: int | None):
    """Stream batched results, with a compatibility fallback for fixed batch sizes."""
    iterator = iter(frames)
    batch_size = max(1, batch_size)
    progress = tqdm(total=total, desc=description, unit="frame", leave=False)
    try:
        while chunk := list(islice(iterator, batch_size)):
            images = [frame for _, frame in chunk]
            try:
                detected = detector.detect_batch(images, batch_size=len(images))
                if not isinstance(detected, list) or len(detected) != len(images):
                    raise TypeError("unexpected NudeNet batch result")
            except Exception:  # noqa: BLE001 - retry unsupported batch shapes one frame at a time
                detected = [_detect_on_frame(detector, cv2, frame) for frame in images]
            for (timestamp, _), hits in zip(chunk, detected):
                yield timestamp, hits or []
            progress.update(len(chunk))
    finally:
        progress.close()


def iter_frame_positions(cv2, cap, offset: float):
    """Grab frames in order, leaving image conversion to the caller."""
    previous = first = None
    count = 0
    while cap.grab():
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000 + offset
        if not math.isfinite(timestamp) or (previous is not None and timestamp < previous):
            raise RuntimeError("Cannot read source frame timestamps.")
        if first is None:
            first = timestamp
        previous = timestamp
        yield count, timestamp
        count += 1
    if count == 0 or (count > 1 and first == previous):
        raise RuntimeError("Cannot read source frames with timestamps.")


def _iter_all_frames(cv2, cap, offset: float):
    for _, timestamp in iter_frame_positions(cv2, cap, offset):
        ok, frame = cap.retrieve()
        if not ok:
            raise RuntimeError(f"Could not decode frame at {timestamp:.3f}s.")
        yield timestamp, frame


def _coarse_times(duration: float, step: float) -> list[tuple[float, float]]:
    step = max(0.05, step)
    count = int(duration / step) + (1 if duration > 0 else 0)
    times = [min(i * step, max(0.0, duration - 0.001)) for i in range(count)]
    return [(t, t) for t in dict.fromkeys(times)]


def _merge_windows(windows: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for start, end in sorted(windows):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def _dense_times(candidates: list[float], duration: float, config: Config) -> list[tuple[float, float]]:
    radius = max(0.1, config.nudity_rescan_window_seconds)
    windows = _merge_windows([
        (max(0.0, t - radius), min(duration, t + radius)) for t in candidates
    ])
    step = 1.0 / max(1.0, config.nudity_rescan_fps)
    values = {round(t, 3) for t in candidates}
    for start, end in windows:
        t = start
        while t <= end:
            values.add(round(t, 3))
            t += step
    return [(t, t) for t in sorted(values) if 0 <= t < duration]


def _confirmed_edl(
    detections_by_time: dict[float, list[dict]],
    duration: float,
    config: Config,
    action: str,
    detector,
    *,
    frame_ends: dict[float, float] | None = None,
    evidence: list[NudityEvidence] | None = None,
) -> EditDecisionList:
    positives = [
        (t, _explicit_hits(detections, config.visual_threshold))
        for t, detections in sorted(detections_by_time.items())
        if _is_explicit(detections, config.visual_threshold)
    ]
    clusters: list[list[tuple[float, list[dict]]]] = []
    for item in positives:
        if clusters and item[0] - clusters[-1][-1][0] <= config.nudity_max_gap_seconds:
            clusters[-1].append(item)
        else:
            clusters.append([item])

    edl = EditDecisionList()
    # Keep the spacing of the Thorough rescan when counting full-rate frames.
    confirmation_interval = 0.1 if config.nudity_all_frames else 0.0
    for cluster in clusters:
        strongest = max(float(hit["score"]) for _, hits in cluster for hit in hits)
        classes = sorted({str(hit["class"]) for _, hits in cluster for hit in hits})
        pad = max(0.0, config.nudity_temporal_padding_seconds)
        backend = (f"{getattr(detector, '_cleancut_model', 'NudeNet')}/"
                   f"{getattr(detector, '_cleancut_backend', 'CPU')}")
        confirmed = False
        for category in classes:
            # Count distinct sampled frames, not boxes. Different anatomy
            # classes and a long series of low scores cannot confirm each other.
            qualified = []
            for t, hits in cluster:
                score = max((float(h["score"]) for h in hits if h["class"] == category), default=0)
                if score >= min(config.nudity_confirmation_threshold, config.nudity_strong_threshold):
                    qualified.append((t, score))
            runs: list[list[tuple[float, float]]] = []
            for hit in qualified:
                if runs and hit[0] - runs[-1][-1][0] <= config.nudity_max_gap_seconds:
                    runs[-1].append(hit)
                else:
                    runs.append([hit])
            for run in runs:
                peak = max(score for _, score in run)
                enough = 0
                previous_confirmation = -math.inf
                for timestamp, score in run:
                    if score < config.nudity_confirmation_threshold:
                        continue
                    if timestamp - previous_confirmation < confirmation_interval - 1e-9:
                        continue
                    enough += 1
                    previous_confirmation = timestamp
                if peak < config.nudity_strong_threshold and enough < max(2, config.nudity_min_confirmations):
                    continue
                confirmed = True
                first, last = run[0][0], run[-1][0]
                end = frame_ends.get(last, last) if frame_ends is not None else last
                edl.add(EditDecision(
                    start=max(0.0, first - pad), end=min(duration, end + pad),
                    action=action, category="nudity", source="visual",
                    reason=(f"confident NudeNet confirmation ({len(run)} samples, max {peak:.2f}, "
                            f"{backend}): {category}"),
                ))
                if evidence is not None:
                    peak_time = max(run, key=lambda hit: hit[1])[0]
                    peak_hit = max(
                        (hit for hit in detections_by_time[peak_time] if hit["class"] == category),
                        key=lambda hit: float(hit["score"]),
                    )
                    evidence.append(NudityEvidence(peak_hit["frame"], peak_time, peak_hit.get("box")))
        if not confirmed and (len(cluster) >= 2 or strongest >= config.nudity_confirmation_threshold):
            first, last = cluster[0][0], cluster[-1][0]
            end = frame_ends.get(last, last) if frame_ends is not None else last
            edl.add(EditDecision(
                start=max(0.0, first - pad), end=min(duration, end + pad),
                action=action, category="nudity", source="visual", accepted=False,
                reason=(f"[needs review; not auto-cut] weak NudeNet evidence ({len(cluster)} samples, "
                        f"max {strongest:.2f}, {backend}): {', '.join(classes)}"),
            ))
            if evidence is not None:
                peak_time, peak_hit = max(
                    ((t, hit) for t, hits in cluster for hit in hits),
                    key=lambda item: float(item[1]["score"]),
                )
                evidence.append(NudityEvidence(peak_hit["frame"], peak_time, peak_hit.get("box")))
    return edl


def _observations_by_frame(samples, fps: float, *, keep_frame_numbers: bool = False) -> dict[float, list[dict]]:
    """Coarse/dense timestamps hitting the same decoded frame count only once."""
    frames: dict[int, list[dict]] = {}
    for t, detections in samples:
        frame = round(t * fps)
        if keep_frame_numbers:
            detections = [{**hit, "frame": frame} for hit in _explicit_hits(detections, 0.0)]
        frames.setdefault(frame, []).extend(detections)
    return {frame / fps: detections for frame, detections in frames.items()}


def _scan_all_frames(
    video_path: Path, config: Config, cv2, detector, evidence: list[NudityEvidence] | None,
) -> EditDecisionList:
    offset, duration, video_end = probe_video_timing(video_path)
    cap = _open_capture(cv2, video_path, source_timestamps=True)
    observations: dict[float, list[dict]] = {}
    frame_ends: dict[float, float] = {}
    previous = None
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
        results = _infer_frames(
            detector, cv2, _iter_all_frames(cv2, cap, offset), config.nudity_batch_size,
            "Nudity scan", total=total,
        )
        for frame_index, (timestamp, detections) in enumerate(results):
            if previous is not None and timestamp > previous and previous in observations:
                frame_ends[previous] = timestamp
            hits = _explicit_hits(detections, config.visual_threshold)
            if hits:
                if evidence is not None:
                    hits = [{**hit, "frame": frame_index} for hit in hits]
                observations.setdefault(timestamp, []).extend(hits)
            previous = timestamp
        if previous is not None and previous in observations:
            fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
            frame_ends[previous] = (
                video_end if video_end is not None and video_end > previous else previous + 1 / fps
            )
    finally:
        cap.release()
    return _confirmed_edl(
        observations, duration, config, config.actions.get("nudity", "cut"), detector,
        frame_ends=frame_ends, evidence=evidence,
    )


def scan_video(
    video_path: Path,
    config: Config,
    shots: list[Shot] | None = None,
    use_cache: bool = True,
    *,
    evidence: list[NudityEvidence] | None = None,
) -> EditDecisionList:
    """Produce temporally confirmed, local NudeNet edit decisions."""
    from cleancut import cache as _cache

    model = resolve_nudity_model(config.nudity_model)
    shot_fingerprint = None
    if shots:
        shot_fingerprint = {
            "n": len(shots),
            "first": (shots[0].start, shots[0].end),
            "last": (shots[-1].start, shots[-1].end),
        }
    first_evidence = len(evidence) if evidence is not None else 0
    cache_hash = _cache.config_hash(
        version="all-frames-1" if config.nudity_all_frames else 3,
        model=model.name,
        threshold=config.visual_threshold,
        strong_threshold=config.nudity_strong_threshold,
        confirmation_threshold=config.nudity_confirmation_threshold,
        sample_seconds=config.visual_sample_seconds,
        rescan_fps=config.nudity_rescan_fps,
        rescan_window=config.nudity_rescan_window_seconds,
        min_confirmations=config.nudity_min_confirmations,
        max_gap=config.nudity_max_gap_seconds,
        temporal_padding=config.nudity_temporal_padding_seconds,
        batch_size=config.nudity_batch_size,
        action=config.actions.get("nudity", "cut"),
        shots=shot_fingerprint,
        **({"frame_evidence": 1} if evidence is not None else {}),
    )
    if use_cache:
        cached = _cache.load(video_path, "nudenet", cache_hash)
        if cached:
            if evidence is not None:
                evidence.extend(NudityEvidence(**item) for item in cached["evidence"])
            return EditDecisionList(
                decisions=[EditDecision(**item) for item in cached.get("decisions", [])]
            )

    cv2, detector = _open_detector(config, model)
    edl = (
        _scan_all_frames(video_path, config, cv2, detector, evidence) if config.nudity_all_frames
        else _scan_sampled_video(video_path, config, cv2, detector, evidence)
    )
    review_count = sum(not d.accepted for d in edl)
    if review_count:
        print(f"Nudity scan: {review_count} low-confidence suggestion(s) need review; "
              "they will not be cut unless accepted.", flush=True)

    if use_cache:
        _cache.save(video_path, "nudenet", cache_hash, {
            "decisions": [asdict(decision) for decision in edl.decisions],
            "evidence": [asdict(item) for item in evidence[first_evidence:]] if evidence is not None else [],
        })
    return edl


def _scan_sampled_video(
    video_path: Path, config: Config, cv2, detector, evidence: list[NudityEvidence] | None,
) -> EditDecisionList:
    cap = _open_capture(cv2, video_path, source_timestamps=evidence is not None)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = frame_count / fps if fps > 0 else 0.0
        coarse = _infer_samples(
            detector, cv2, cap, fps,
            _coarse_times(duration, config.visual_sample_seconds),
            config.nudity_batch_size, "Nudity scan",
        )
    finally:
        cap.release()

    candidates = [
        t for t, detections in coarse
        if _is_explicit(detections, config.visual_threshold)
    ]
    if not candidates:
        return EditDecisionList()
    cap = _open_capture(cv2, video_path, source_timestamps=evidence is not None)
    try:
        dense = _infer_samples(
            detector, cv2, cap, fps, _dense_times(candidates, duration, config),
            config.nudity_batch_size, "Confirming nudity",
        )
    finally:
        cap.release()
    # Keep coarse observations if the rescan decodes a neighboring frame.
    observations = _observations_by_frame([*dense, *coarse], fps, keep_frame_numbers=evidence is not None)
    return _confirmed_edl(
        observations, duration, config, config.actions.get("nudity", "cut"), detector,
        evidence=evidence,
    )
