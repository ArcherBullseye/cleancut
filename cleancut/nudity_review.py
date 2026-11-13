"""Compare NudeNet candidates with an Ollama vision model and look for misses."""

from __future__ import annotations

import json
import math
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import asdict, replace
from pathlib import Path

from tqdm import tqdm

from cleancut import cache
from cleancut.config import Config
from cleancut.edl import EditDecision
from cleancut.llm_utils import chat_for_json, make_ollama_client, preflight_ollama, strip_to_json
from cleancut.probe import probe_video_timing
from cleancut.visual import NudityEvidence, _open_capture, iter_frame_positions

VERDICTS = ("visible", "absent", "uncertain")
SCHEMA = {
    "type": "object",
    "properties": {"verdict": {"type": "string", "enum": list(VERDICTS)}},
    "required": ["verdict"],
}
PROMPT = (
    "Judge visible nudity in the first image. Look for exposed female breasts, genitals, "
    "buttocks, or anus. Use any other images as context for that first image. "
    "If a crop is provided, judge the region it shows in the first image. "
    "Clothing, underwear, cleavage, a bare male chest, shadows, and skin-colored objects "
    "alone are insufficient. Judge what is actually visible, even if it is brief or nonsexual. "
    "Return a JSON object with verdict visible, absent, or uncertain. Use uncertain if the "
    "image is too unclear to tell."
)
SAMPLE_SECONDS = 1.0
CHECKPOINT_SAMPLES = 120


def _image_bytes(cv2, image) -> bytes:
    height, width = image.shape[:2]
    scale = 768 / max(height, width)
    if scale < 1:
        image = cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))),
                           interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError("Could not encode an image for Ollama.")
    return encoded.tobytes()


def _crop(image, box: list[int] | None):
    if not box:
        return None
    x, y, width, height = box
    if width <= 0 or height <= 0:
        return None
    image_height, image_width = image.shape[:2]
    x_pad, y_pad = width // 4, height // 4
    left, right = max(0, x - x_pad), min(image_width, x + width + x_pad)
    top, bottom = max(0, y - y_pad), min(image_height, y + height + y_pad)
    if left >= right or top >= bottom:
        return None
    crop = image[top:bottom, left:right]
    return crop if crop.size else None


def _frame_image(cap, timestamp: float):
    ok, image = cap.retrieve()
    if not ok:
        raise RuntimeError(f"Could not read the frame at {timestamp:.3f}s.")
    return image


def _classify(client, model: str, images: list[bytes], context: str) -> str:
    for attempt in (1, 2):
        try:
            text = chat_for_json(
                client, model=model, format=SCHEMA, options={"temperature": 0},
                messages=[{"role": "system", "content": PROMPT},
                          {"role": "user", "content": context, "images": images}],
            )
            verdict = json.loads(strip_to_json(text))["verdict"]
            if verdict not in VERDICTS:
                raise ValueError("Unknown verdict")
            return verdict
        except Exception as exc:
            if attempt == 2:
                raise RuntimeError(f"Ollama vision model {model!r} failed: {exc}") from exc
    raise AssertionError("Classification must return or raise")


def _candidate_images(cv2, video: Path, offset: float, evidence: list[NudityEvidence]) -> Iterator:
    """Yield exact detector frames with their nearest decoded neighbors."""
    targets: dict[int, list[int]] = {}
    for i, item in enumerate(evidence):
        targets.setdefault(item.frame, []).append(i)
    wanted = {frame + delta for frame in targets for delta in (-1, 0, 1)}
    last = max(targets)
    images = deque(maxlen=3)
    cap = _open_capture(cv2, video, source_timestamps=True)
    try:
        for frame, timestamp in iter_frame_positions(cv2, cap, offset):
            if frame not in wanted:
                continue
            images.append((frame, timestamp, _frame_image(cap, timestamp)))
            if frame - 1 in targets:
                yield targets.pop(frame - 1), list(images)
            if frame > last:
                break
        if last in targets and images and images[-1][0] == last:
            yield targets.pop(last), list(images)
        if targets:
            raise RuntimeError("Could not recover every flagged NudeNet frame.")
    finally:
        cap.release()


def _verify(cv2, video, offset, evidence, client, model) -> list[tuple[str, float]]:
    results: dict[int, tuple[str, float]] = {}
    if evidence:
        with tqdm(total=len(evidence), desc="Checking NudeNet candidates", unit="candidate") as progress:
            for indices, images in _candidate_images(cv2, video, offset, evidence):
                for index in indices:
                    item = evidence[index]
                    frame, timestamp, image = next(row for row in images if row[0] == item.frame)
                    encoded = [_image_bytes(cv2, image)]
                    context = "Image 1 is the frame to judge."
                    crop = _crop(image, item.box)
                    if crop is not None:
                        encoded.append(_image_bytes(cv2, crop))
                        context += " Image 2 is a crop of that frame."
                    for neighbor, _, neighbor_image in images:
                        if abs(neighbor - frame) == 1:
                            encoded.append(_image_bytes(cv2, neighbor_image))
                    if len(encoded) > 1 + (crop is not None):
                        context += " The final images are nearby frames."
                    results[index] = (_classify(client, model, encoded, context), timestamp)
                    progress.update()
    return [results[index] for index in range(len(evidence))]


def _search(
    cv2, video, offset, duration, client, model, *,
    next_sample: float, positives: list[float], checkpoint: Callable[[float, list[float]], None],
) -> list[float]:
    """Sample the full timeline independently, including shots with dialogue."""
    cap = _open_capture(cv2, video, source_timestamps=True)
    processed = 0
    first_sample = min(SAMPLE_SECONDS / 2, duration / 2)
    try:
        with tqdm(total=duration, desc="Looking for additional nudity", unit="s") as progress:
            progress.update(max(0.0, min(duration, next_sample - SAMPLE_SECONDS)))
            for _, timestamp in iter_frame_positions(cv2, cap, offset):
                if timestamp < next_sample:
                    continue
                image = _image_bytes(cv2, _frame_image(cap, timestamp))
                if _classify(client, model, [image], "Judge this frame.") == "visible":
                    positives.append(timestamp)
                next_sample += (math.floor((timestamp - next_sample) / SAMPLE_SECONDS) + 1) * SAMPLE_SECONDS
                processed += 1
                if processed % CHECKPOINT_SAMPLES == 0:
                    checkpoint(next_sample, positives)
                progress.update(max(0.0, min(timestamp, duration) - progress.n))
            # Very short clips can end before the first scheduled sample.
            if not processed and next_sample == first_sample:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                if not cap.grab():
                    raise RuntimeError("Could not sample the source video.")
                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000 + offset
                image = _image_bytes(cv2, _frame_image(cap, timestamp))
                if _classify(client, model, [image], "Judge this frame.") == "visible":
                    positives.append(timestamp)
            progress.update(duration - progress.n)
    finally:
        cap.release()
    return positives


def review_nudity(
    source: Path, detector_video: Path,
    decisions: list[EditDecision], evidence: list[NudityEvidence], config: Config,
) -> list[EditDecision]:
    """Annotate NudeNet decisions and return extra, unselected suggestions."""
    if len(decisions) != len(evidence):
        raise RuntimeError("NudeNet frame evidence is missing. Scan the original video again.")
    import cv2

    model = config.vlm_model
    source_offset, duration, _ = probe_video_timing(source)
    model_key = {"version": 1, "model": model, "host": config.llm_host, "prompt": PROMPT}
    check_key = cache.config_hash(**model_key, evidence=[asdict(item) for item in evidence])
    search_key = cache.config_hash(**model_key, cadence=SAMPLE_SECONDS)
    checks = cache.load(detector_video, "nudity_verification", check_key)
    search = cache.load(source, "nudity_search", search_key)
    if checks is None or search is None:
        print(f"Ollama nudity review using {model}", flush=True)
        client = make_ollama_client(config.llm_host, timeout=120.0)
        preflight_ollama(client, model)
        if checks is None:
            detector_offset = source_offset
            if detector_video != source:
                detector_offset, _, _ = probe_video_timing(detector_video)
            checks = {"verdicts": _verify(cv2, detector_video, detector_offset, evidence, client, model)}
            cache.save(detector_video, "nudity_verification", check_key, checks)
        if search is None:
            def save_progress(next_sample: float, timestamps: list[float]) -> None:
                cache.save(
                    source, "nudity_search_partial", search_key,
                    {"next_sample": next_sample, "timestamps": timestamps},
                )

            partial = cache.load(source, "nudity_search_partial", search_key) or {}
            search = {"timestamps": _search(
                cv2, source, source_offset, duration, client, model,
                next_sample=partial.get("next_sample", min(SAMPLE_SECONDS / 2, duration / 2)),
                positives=partial.get("timestamps", []), checkpoint=save_progress,
            )}
            cache.save(source, "nudity_search", search_key, search)

    action = config.actions.get("nudity", "cut")
    notes = {"visible": "Ollama agrees", "absent": "Ollama disagrees", "uncertain": "Ollama is uncertain"}
    reviewed = []
    for decision, (verdict, timestamp) in zip(decisions, checks["verdicts"], strict=True):
        prefix = "[needs review; not auto-cut] " if not decision.accepted else ""
        reason = decision.reason.removeprefix("[needs review; not auto-cut] ")
        note = f"[{notes[verdict]}] {model} at {timestamp:.3f}s"
        reviewed.append(replace(decision, reason=f"{prefix}{note}. {reason}"))

    for timestamp in search["timestamps"]:
        if any(d.start <= timestamp < d.end for d in decisions):
            continue
        start, end = max(0.0, timestamp - 0.5), min(duration, timestamp + 0.5)
        if start < end:
            reviewed.append(EditDecision(
                start=start, end=end, action=action, category="nudity", accepted=False,
                source="ollama-nudity", reason=f"[needs review; not auto-cut] {model} flagged a frame at {timestamp:.3f}s.",
            ))
    return reviewed
