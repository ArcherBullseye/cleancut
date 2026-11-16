"""Classify sequences of frames and cut complete shots showing sexual activity."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from tqdm import tqdm

from cleancut import cache
from cleancut.config import Config
from cleancut.edl import EditDecision
from cleancut.llm_utils import make_ollama_client, preflight_ollama
from cleancut.nudity_review import _classify, _frame_image, _image_bytes
from cleancut.probe import probe_video_timing
from cleancut.scenes import Shot
from cleancut.visual import _open_capture, iter_frame_positions

PROMPT = (
    "Judge the visible activity in these chronological frames from the same shot for a film editor. "
    "Return active if the images clearly depict people engaged in a sex act. This includes clearly "
    "depicted intercourse or oral sex even with clothing or partial framing. Return continuation "
    "only if this looks like an immediate reaction or insert in an ongoing act whose action is "
    "out of view. Return absent for nudity by itself, bathing, changing, medical care, kissing, "
    "embracing or simply sharing a bed. Use uncertain when the depicted action is ambiguous. "
    "Return one JSON verdict."
)
SCHEMA = {
    "type": "object",
    "properties": {"verdict": {"type": "string", "enum": ["active", "continuation", "absent", "uncertain"]}},
    "required": ["verdict"],
}
SAMPLE_INTERVAL = 0.75
FRAMES_PER_REQUEST = 4


def _shot_verdicts(
    video: Path, shots: list[Shot], config: Config, *,
    previous: list[str], checkpoint: Callable[[list[str]], None],
) -> list[str]:
    import cv2

    offset, _, _ = probe_video_timing(video)
    client = make_ollama_client(config.llm_host, timeout=120.0)
    preflight_ollama(client, config.vlm_model)
    cap = _open_capture(cv2, video, source_timestamps=True)
    frames = iter(iter_frame_positions(cv2, cap, offset))
    verdicts = list(previous)
    try:
        position = next(frames, None)
        remaining = shots[len(verdicts):]
        for shot in tqdm(remaining, desc="Checking for sexual activity", unit="shot", leave=False,
                         initial=len(verdicts), total=len(shots)):
            results = []
            images = []
            next_sample = shot.start
            while position is not None and position[1] < shot.end:
                _, timestamp = position
                if timestamp >= next_sample:
                    images.append(_image_bytes(cv2, _frame_image(cap, timestamp)))
                    next_sample = timestamp + SAMPLE_INTERVAL
                if len(images) == FRAMES_PER_REQUEST:
                    results.append(_judge(client, config, images))
                    images = []
                    if results[-1] == "active":
                        break
                position = next(frames, None)
            if images:
                results.append(_judge(client, config, images))
            verdicts.append(next((v for v in SCHEMA["properties"]["verdict"]["enum"] if v in results), "absent"))
            if len(verdicts) % 30 == 0:
                checkpoint(verdicts)
    finally:
        cap.release()
    return verdicts


def _judge(client, config: Config, images: list[bytes]) -> str:
    return _classify(
        client, config.vlm_model, images, "Judge the activity in this shot.", prompt=PROMPT, schema=SCHEMA,
    )


def scan_sexual_activity(video: Path, shots: list[Shot], config: Config) -> list[EditDecision]:
    """Include consecutive shots of the same activity in one scene cut."""
    if not shots:
        raise RuntimeError("Shot boundaries are required to cut whole sex scenes.")
    key = cache.config_hash(
        version=1, model=config.vlm_model, host=config.llm_host, prompt=PROMPT,
        interval=SAMPLE_INTERVAL, batch=FRAMES_PER_REQUEST,
        shots=[(shot.start, shot.end) for shot in shots],
    )
    saved = cache.load(video, "sexual_activity", key)
    if saved is None:
        partial = cache.load(video, "sexual_activity_partial", key) or {}

        def save_progress(verdicts: list[str]) -> None:
            cache.save(video, "sexual_activity_partial", key, {"verdicts": verdicts})

        verdicts = _shot_verdicts(video, shots, config, previous=partial.get("verdicts", []),
                                  checkpoint=save_progress)
        cache.save(video, "sexual_activity", key, {"verdicts": verdicts})
    else:
        verdicts = saved["verdicts"]

    decisions = []
    group: list[tuple[Shot, str]] = []
    for shot, verdict in zip([*shots, Shot(0, 0)], [*verdicts, "absent"], strict=True):
        if group and (verdict not in {"active", "continuation"} or shot.start > group[-1][0].end + 0.01):
            accepted = any(value == "active" for _, value in group)
            decisions.append(EditDecision(
                group[0][0].start, group[-1][0].end, "cut", "sex", source="sexual-activity", accepted=accepted,
                reason=("Ollama detected sexual activity across this scene." if accepted else
                        "[needs review] Ollama saw a possible continuation of sexual activity."),
            ))
            group = []
        if verdict in {"active", "continuation"}:
            group.append((shot, verdict))
        elif verdict == "uncertain":
            decisions.append(EditDecision(
                shot.start, shot.end, "cut", "sex", source="sexual-activity", accepted=False,
                reason="[needs review] Ollama was uncertain whether this shot depicted sexual activity.",
            ))
    return decisions
