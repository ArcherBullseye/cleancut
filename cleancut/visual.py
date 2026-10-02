"""Local, two-pass nudity detection using NudeNet.

The first pass scans the full video at a configurable cadence. Every possible
explicit-content hit opens a short dense-rescan window. A window is accepted
when it contains repeated hits or one high-confidence hit. This catches brief
nudity inside long shots without letting isolated borderline detections create
cuts.
"""

from __future__ import annotations

import platform
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path

from tqdm import tqdm

from cleancut.config import Config
from cleancut.edl import EditDecision, EditDecisionList
from cleancut.nudity_model import NudityModel, resolve_nudity_model
from cleancut.scenes import Shot

EXPLICIT_CLASSES = {
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "BUTTOCKS_EXPOSED",
    "ANUS_EXPOSED",
}


def _explicit_hits(detections, threshold: float) -> list[dict]:
    return [
        d for d in detections
        if d.get("class") in EXPLICIT_CLASSES
        and float(d.get("score", 0)) >= threshold
    ]


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


def _open_capture(cv2, video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
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
    """Run batched inference, with a compatibility fallback."""
    output: list[tuple[float, list[dict]]] = []
    iterator = iter(_iter_sampled_frames(cap, fps, samples))
    progress = tqdm(total=len(samples), desc=description, unit="frame", leave=False)
    try:
        while True:
            chunk = []
            for _ in range(max(1, batch_size)):
                try:
                    chunk.append(next(iterator))
                except StopIteration:
                    break
            if not chunk:
                break
            frames = [frame for _, frame in chunk]
            try:
                detected = detector.detect_batch(frames, batch_size=len(frames))
                if not isinstance(detected, list) or len(detected) != len(frames):
                    raise TypeError("unexpected NudeNet batch result")
            except Exception:  # noqa: BLE001 - retry unsupported batch shapes one frame at a time
                detected = [_detect_on_frame(detector, cv2, frame) for frame in frames]
            output.extend((chunk[i][0], detected[i] or []) for i in range(len(chunk)))
            progress.update(len(chunk))
    finally:
        progress.close()
    return output


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
    for cluster in clusters:
        scores = [float(hit.get("score", 0)) for _, hits in cluster for hit in hits]
        if not scores:
            continue
        strongest = max(scores)
        if (
            strongest < config.nudity_strong_threshold
            and len(cluster) < max(1, config.nudity_min_confirmations)
        ):
            continue
        classes = sorted({str(hit["class"]) for _, hits in cluster for hit in hits})
        pad = max(0.0, config.nudity_temporal_padding_seconds)
        edl.add(EditDecision(
            start=max(0.0, cluster[0][0] - pad),
            end=min(duration, cluster[-1][0] + pad),
            action=action,
            category="nudity",
            reason=(
                f"temporal NudeNet confirmation ({len(cluster)} samples, "
                f"max {strongest:.2f}, {getattr(detector, '_cleancut_model', 'NudeNet')}/"
                f"{getattr(detector, '_cleancut_backend', 'CPU')}): {', '.join(classes)}"
            ),
            source="visual",
        ))
    return edl


def scan_video(
    video_path: Path,
    config: Config,
    shots: list[Shot] | None = None,
    use_cache: bool = True,
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
    cache_hash = _cache.config_hash(
        version=2,
        model=model.name,
        threshold=config.visual_threshold,
        strong_threshold=config.nudity_strong_threshold,
        sample_seconds=config.visual_sample_seconds,
        rescan_fps=config.nudity_rescan_fps,
        rescan_window=config.nudity_rescan_window_seconds,
        min_confirmations=config.nudity_min_confirmations,
        batch_size=config.nudity_batch_size,
        action=config.actions.get("nudity", "cut"),
        shots=shot_fingerprint,
    )
    if use_cache:
        cached = _cache.load(video_path, "nudenet", cache_hash)
        if cached:
            return EditDecisionList(
                decisions=[EditDecision(**item) for item in cached.get("decisions", [])]
            )

    cv2, detector = _open_detector(config, model)
    cap = _open_capture(cv2, video_path)
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
        edl = EditDecisionList()
    else:
        cap = _open_capture(cv2, video_path)
        try:
            dense = _infer_samples(
                detector, cv2, cap, fps, _dense_times(candidates, duration, config),
                config.nudity_batch_size, "Confirming nudity",
            )
        finally:
            cap.release()
        # Keep coarse observations too: a high-confidence hit must not disappear
        # merely because the decoder returns a neighboring frame during rescan.
        observations = {round(t, 3): detections for t, detections in dense}
        for t, detections in coarse:
            if _is_explicit(detections, config.visual_threshold):
                observations[round(t, 3)] = detections
        edl = _confirmed_edl(
            observations, duration, config,
            config.actions.get("nudity", "cut"), detector,
        )

    if use_cache:
        _cache.save(video_path, "nudenet", cache_hash, {
            "decisions": [asdict(decision) for decision in edl.decisions],
        })
    return edl
