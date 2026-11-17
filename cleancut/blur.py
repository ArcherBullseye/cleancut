"""Build moving blur masks on the source timeline for accepted nudity edits."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import chain
from pathlib import Path

from cleancut import cache
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

Box = tuple[int, int, int, int]


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


def _frames_in_ranges(
    cv2, cap, offset: float, video_end: float, ranges: list[Range],
    events: dict[float, list[Box]], full_frame: Box,
):
    spans = iter(ranges)
    current = next(spans, None)
    previous = previous_when = None
    previous_image = None
    was_selected = False
    fps = cap.get(cv2.CAP_PROP_FPS)
    lookahead = 2 / fps if fps > 0 else 0.0

    positions = chain(iter_frame_positions(cv2, cap, offset), [(None, video_end)])
    for index, timestamp in positions:
        if previous is not None and current and not was_selected and previous < current.start < timestamp:
            # A displayed frame extends to the next timestamp. This matters for
            # ranges shorter than a frame and for a long hold in a VFR video.
            if previous_image is not None:
                yield previous_when, previous_image
            else:
                events[previous_when] = [full_frame]
            was_selected = True
        if index is None:
            break
        while current is not None and timestamp >= current.end:
            current = next(spans, None)
        selected = current is not None and timestamp >= current.start
        # Place the command between timestamps to allow for decoder rounding.
        when = max(0.0, (previous + timestamp) / 2 if previous is not None else timestamp - 0.001)
        repeated = timestamp == previous
        if repeated:
            when = previous_when
        if selected:
            ok, image = cap.retrieve()
            if not ok:
                raise RuntimeError(f"Could not read blur frame at {timestamp:.3f}s.")
            yield when, image
        elif was_selected:
            events[when] = []
        if current is None:
            break
        # Keep the frame just before a range when possible. Earlier frames can
        # stay unconverted, including gaps between edits in long movies.
        previous_image = None
        if not selected and not repeated and current.start - timestamp <= lookahead:
            ok, image = cap.retrieve()
            if ok:
                previous_image = image
        was_selected = selected
        previous = timestamp
        previous_when = when


def _boxes(hits, width: int, height: int) -> list[Box]:
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


def _file_identity(path: Path) -> tuple[int, ...]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def prepare_blur(
    video: Path, ranges: list[Range], config: Config, commands: Path, *, use_cache: bool = True,
) -> BlurFilter:
    """Locate all regions in selected frames. Cover the full frame when localization fails."""
    model = resolve_nudity_model(config.nudity_model)
    threshold = min(config.visual_threshold, 0.35)
    ranges = normalize_cuts(ranges)
    source_identity = _file_identity(video)
    key = None
    if use_cache and isinstance(model.path, Path) and model.path.is_file():
        key = cache.config_hash(
            version=1, model=model.name, resolution=model.resolution,
            weights=_file_identity(model.path), source=source_identity,
            ranges=[(r.start, r.end) for r in ranges], threshold=threshold,
            batch_size=config.nudity_batch_size, coreml=config.nudity_coreml,
        )
        saved = cache.load(video, "blur", key)
        if saved:
            if _file_identity(video) != source_identity:
                raise RuntimeError("Source video changed while loading blur regions.")
            regions = saved["regions"]
            width, height = saved["dimensions"]
            commands.write_text(saved["commands"], encoding="utf-8")
            print("Reusing cached nudity blur regions.", flush=True)
            return BlurFilter(commands, regions, max(12.0, min(width, height) / 16), (width, height))

    offset, duration, video_end = probe_video_timing(video)
    ranges = normalize_cuts(ranges, duration)
    cv2, detector = _open_detector(config, model)
    cap = _open_capture(cv2, video, source_timestamps=True)
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            raise RuntimeError("Could not read source dimensions for nudity blur.")
        full_frame = (0, 0, width, height)
        events: dict[float, list[Box]] = {}
        results = _infer_frames(
            detector, cv2,
            _frames_in_ranges(cv2, cap, offset, video_end or duration, ranges, events, full_frame),
            config.nudity_batch_size,
            "Locating blur regions", total=None,
        )
        for when, detections in results:
            boxes = _boxes(_explicit_hits(detections, threshold), width, height)
            # Frames sharing a timestamp need a mask for the union of detections.
            regions = events.setdefault(when, [])
            if not boxes or full_frame in boxes or full_frame in regions:
                events[when] = [full_frame]
            else:
                regions.extend(box for box in boxes if box not in regions)
    finally:
        cap.release()
    regions = max(map(len, events.values()), default=0)
    if not regions:
        raise RuntimeError("No source frames were found inside the selected blur ranges.")
    fallback = sum(boxes == [full_frame] for boxes in events.values())
    if fallback:
        print(f"Nudity blur will cover {fallback} whole frame(s).", flush=True)

    previous_boxes: list[Box] = []
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
    if _file_identity(video) != source_identity:
        raise RuntimeError("Source video changed while locating blur regions.")
    if key is not None:
        try:
            cache.save(video, "blur", key, {
                "commands": commands.read_text(encoding="utf-8"), "regions": regions, "dimensions": (width, height),
            })
        except OSError as exc:
            print(f"Could not cache nudity blur regions: {exc}", flush=True)
    return BlurFilter(commands, regions, max(12.0, min(width, height) / 16), (width, height))
