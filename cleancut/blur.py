"""Build moving blur masks on the source timeline for accepted nudity edits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cleancut.config import Config
from cleancut.editor_ranges import Range, normalize_cuts
from cleancut.nudity_model import resolve_nudity_model
from cleancut.probe import probe_video_timing
from cleancut.visual import (
    _explicit_hits,
    _infer_frames,
    _open_capture,
    _open_detector,
    iter_frame_positions,
)


@dataclass(frozen=True)
class BlurFilter:
    commands: Path
    regions: int
    sigma: float
    dimensions: tuple[int, int]

    def graph(self, output: str, encoder: str) -> str:
        overlay_format = "yuv420p10" if encoder in {"hevc_videotoolbox", "libx265"} else "yuv420"
        boxes = ",".join(
            f"drawbox@region{i}=x=iw:y=0:w=1:h=1:color=white:t=fill" for i in range(self.regions)
        )
        return (
            "[0:v:0]sendcmd=f=blur.commands,split=3[sharp][soft][maskin];"
            f"[soft]gblur@soften=sigma={self.sigma:.2f}:steps=3:enable=0[blurred];"
            f"[maskin]format=gray,lut=c0=0,{boxes}[mask];"
            f"[blurred][mask]alphamerge[regionblur];[sharp][regionblur]overlay=format={overlay_format}{output}"
        )


def _frames_in_ranges(cv2, cap, offset: float, ranges: list[Range], events: dict):
    spans = iter(ranges)
    current = next(spans, None)
    previous = None
    was_selected = False
    for _, timestamp in iter_frame_positions(cv2, cap, offset):
        while current is not None and timestamp >= current.end:
            current = next(spans, None)
        selected = current is not None and timestamp >= current.start
        # Place the command between timestamps to allow for decoder rounding.
        when = max(0.0, (previous + timestamp) / 2 if previous is not None else timestamp - 0.001)
        if selected:
            ok, image = cap.retrieve()
            if not ok:
                raise RuntimeError(f"Could not read blur frame at {timestamp:.3f}s.")
            yield when, image
        elif was_selected:
            events[when] = []
        if current is None:
            break
        was_selected = selected
        previous = timestamp


def _boxes(hits, width: int, height: int) -> list[tuple[int, int, int, int]]:
    boxes = []
    for hit in hits:
        box = hit.get("box")
        if not box or len(box) != 4:
            continue
        x, y, w, h = map(int, box)
        margin_x, margin_y = max(4, w // 3), max(4, h // 3)
        left, top = max(0, x - margin_x), max(0, y - margin_y)
        right, bottom = min(width, x + w + margin_x), min(height, y + h + margin_y)
        if w > 0 and h > 0 and left < right and top < bottom:
            boxes.append((left, top, right - left, bottom - top))
    return boxes


def prepare_blur(video: Path, ranges: list[Range], config: Config, commands: Path) -> BlurFilter:
    """Locate all regions in selected frames. Cover the full frame when localization fails."""
    offset, duration, _ = probe_video_timing(video)
    ranges = normalize_cuts(ranges, duration)
    model = resolve_nudity_model(config.nudity_model)
    cv2, detector = _open_detector(config, model)
    cap = _open_capture(cv2, video, source_timestamps=True)
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            raise RuntimeError("Could not read source dimensions for nudity blur.")
        events: dict[float, list[tuple[int, int, int, int]]] = {}
        results = _infer_frames(
            detector, cv2, _frames_in_ranges(cv2, cap, offset, ranges, events), config.nudity_batch_size,
            "Locating blur regions", total=None,
        )
        fallback = 0
        for when, detections in results:
            boxes = _boxes(_explicit_hits(detections, min(config.visual_threshold, 0.35)), width, height)
            if not boxes:
                boxes = [(0, 0, width, height)]
                fallback += 1
            events[when] = boxes
    finally:
        cap.release()
    regions = max(map(len, events.values()), default=0)
    if not regions:
        raise RuntimeError("No source frames were found inside the selected blur ranges.")
    if fallback:
        print(f"Nudity blur will cover {fallback} whole frame(s) where no region was located.", flush=True)

    previous_boxes: list[tuple[int, int, int, int]] = []
    with commands.open("w", encoding="utf-8") as script:
        for time, boxes in sorted(events.items()):
            changes = []
            if bool(boxes) != bool(previous_boxes):
                changes.append(f"gblur@soften enable {int(bool(boxes))}")
            for i, box in enumerate(boxes):
                if i < len(previous_boxes) and box == previous_boxes[i]:
                    continue
                changes.extend(f"drawbox@region{i} {key} {value}" for key, value in zip("xywh", box))
            changes.extend(f"drawbox@region{i} x iw" for i in range(len(boxes), len(previous_boxes)))
            if changes:
                script.write(f"{time:.8f} {', '.join(changes)};\n")
            previous_boxes = boxes
    return BlurFilter(commands, regions, max(12.0, min(width, height) / 16), (width, height))
