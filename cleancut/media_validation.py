"""Post-render validation for catching truncated or corrupt video output."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from cleancut.probe import probe_duration, probe_streams, video_stream


def validate_media(
    path: Path,
    *,
    expected_duration: float | None = None,
    expected_dimensions: tuple[int, int] | None = None,
    mode: str = "quick",
) -> None:
    """Raise RuntimeError when a rendered file is incomplete or undecodable.

    ``quick`` checks container metadata, the video stream, dimensions, and
    duration. ``full`` additionally decodes the complete first video/audio
    streams with ffmpeg's error-to-failure mode, which catches damaged frames
    that a metadata probe cannot see.
    """
    if mode == "none":
        return
    if mode not in {"quick", "full"}:
        raise ValueError(f"unknown render validation mode: {mode}")
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError("render validation failed: output is missing or empty")

    try:
        duration = probe_duration(path)
        stream = video_stream(probe_streams(path))
    except Exception as e:
        raise RuntimeError(f"render validation failed: ffprobe could not read output: {e}") from e
    if duration <= 0:
        raise RuntimeError("render validation failed: output duration is zero")
    if stream is None:
        raise RuntimeError("render validation failed: output has no video stream")
    if not stream.width or not stream.height:
        raise RuntimeError("render validation failed: output has invalid video dimensions")
    if expected_dimensions and (stream.width, stream.height) != expected_dimensions:
        raise RuntimeError(
            "render validation failed: dimensions changed "
            f"(expected {expected_dimensions[0]}x{expected_dimensions[1]}, "
            f"got {stream.width}x{stream.height})"
        )

    # Apple decoders use the codec declaration to decide whether to accept a
    # stream before decoding it. Guard the two metadata mistakes that have
    # caused otherwise-decodable CleanCut output to fail in QuickTime/Safari.
    if path.suffix.lower() in {".mp4", ".m4v", ".mov"}:
        if stream.codec_name in {"hevc", "h265"} and stream.codec_tag_string != "hvc1":
            raise RuntimeError(
                "render validation failed: HEVC in an Apple container must use the hvc1 tag"
            )
        if stream.level == 255 or (stream.codec_name == "h264" and (stream.level or 0) > 52):
            raise RuntimeError(
                f"render validation failed: incompatible {stream.codec_name} codec level "
                f"{stream.level}"
            )

    if expected_duration is not None and expected_duration > 0:
        # Containers and AAC priming can differ by a few frames. Long sources
        # get at most three seconds of tolerance, never a percentage-sized gap.
        tolerance = max(1.0, min(3.0, expected_duration * 0.001))
        if abs(duration - expected_duration) > tolerance:
            raise RuntimeError(
                "render validation failed: duration mismatch "
                f"(expected {expected_duration:.3f}s, got {duration:.3f}s)"
            )

    if mode == "full":
        if not shutil.which("ffmpeg"):
            raise RuntimeError("render validation failed: ffmpeg not found")
        try:
            subprocess.run(
                [
                    "ffmpeg", "-nostdin", "-v", "error", "-xerror",
                    "-i", str(path),
                    "-map", "0:v:0", "-map", "0:a:0?",
                    "-f", "null", "-",
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            detail = (e.stderr or "unknown decode error").strip()
            raise RuntimeError(f"render validation failed during full decode: {detail}") from e
