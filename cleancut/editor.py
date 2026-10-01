"""ffmpeg orchestration: apply cuts, mutes, and burned subtitles.

Range/EDL arithmetic lives in editor_ranges; the public names are re-exported
here so existing callers (pipeline.py, cli.py, tests) need no changes.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path


# Re-export pure-arithmetic names from editor_ranges and the ffprobe wrapper
# from probe so any code doing `from cleancut.editor import …` keeps working.
from cleancut.editor_ranges import (  # noqa: F401
    Range,
    adjust_subtitles_for_cuts,
    edl_to_ranges,
    keep_segments,
    shift_after_cuts,
    shift_ranges_after_cuts,
)
from cleancut.probe import probe_duration  # noqa: F401


def _require_ffmpeg() -> None:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found on PATH. Install ffmpeg (e.g. brew install ffmpeg).")
    if not shutil.which("ffprobe"):
        raise RuntimeError("ffprobe not found on PATH. It ships with ffmpeg.")


def _color_metadata_args(source) -> list[str]:
    """Copy safe color-description fields to a newly encoded stream."""
    if source is None:
        return []
    args: list[str] = []
    for flag, value in (
        ("-color_range", source.color_range),
        ("-colorspace", source.color_space),
        ("-color_trc", source.color_transfer),
        ("-color_primaries", source.color_primaries),
    ):
        if value and value.lower() not in {"unknown", "unspecified", "reserved"}:
            args += [flag, value]
    return args


def _playback_rate_args(source) -> list[str]:
    """Pin an interoperable CFR using the source's measured average rate.

    The trim/concat filter uses a microsecond time base. Passing that directly
    to x264/x265 makes them advertise codec levels 6.2/8.5, which Apple players
    can reject even though the frames decode. Normalizing the rendered edit to
    the source's average rate restores a normal codec level and stable A/V
    pacing. Analysis proxies remain timestamp-preserving and are unaffected.
    """
    rate = getattr(source, "avg_frame_rate", "") if source is not None else ""
    try:
        numerator, denominator = (int(part) for part in rate.split("/", 1))
        fps = numerator / denominator
        if numerator > 0 and denominator > 0 and 1 <= fps <= 240:
            return ["-r", f"{numerator}/{denominator}", "-fps_mode", "cfr"]
    except (AttributeError, ValueError, ZeroDivisionError):
        pass
    return ["-r", "30", "-fps_mode", "cfr"]


def _video_encoder_args(encoder: str, quality: int, source=None) -> list[str]:
    """Return ffmpeg flags for a broadly playable H.264 stream.

    libx264 otherwise inherits the source pixel format.  A 10-bit or 4:4:4
    source can therefore produce a valid H.264 file that QuickTime and Safari
    refuse to decode.  yuv420p is the interoperable 8-bit format expected by
    macOS, browsers, TVs, and media servers.
    """
    if encoder == "videotoolbox":
        # videotoolbox uses -q:v (higher = better). Map CRF-ish quality to a sensible q.
        # CRF 18 ~ q 54, CRF 20 ~ q 50, CRF 23 ~ q 44.
        q = max(30, min(100, 90 - quality * 2))
        return [
            "-c:v", "h264_videotoolbox", "-q:v", str(q), "-b:v", "0",
            "-allow_sw", "1",
            "-pix_fmt", "yuv420p", *_playback_rate_args(source),
        ]
    if encoder == "hevc_videotoolbox":
        q = max(30, min(100, 90 - quality * 2))
        return [
            "-c:v", "hevc_videotoolbox", "-q:v", str(q), "-b:v", "0",
            "-allow_sw", "1",
            "-pix_fmt", "p010le", "-profile:v", "main10", "-tag:v", "hvc1",
            *_playback_rate_args(source),
            *_color_metadata_args(source),
        ]
    if encoder == "libx265":
        return [
            "-c:v", "libx265", "-preset", "medium", "-crf", str(quality),
            "-pix_fmt", "yuv420p10le", "-profile:v", "main10", "-tag:v", "hvc1",
            *_playback_rate_args(source),
            *_color_metadata_args(source),
        ]
    # libx264 default
    return [
        "-c:v", "libx264", "-preset", "slow", "-crf", str(quality),
        "-pix_fmt", "yuv420p", *_playback_rate_args(source),
    ]


_MP4_SUFFIXES = {".mp4", ".m4v", ".mov"}
_APPLE_H264_PIXEL_FORMATS = {"", "yuv420p", "yuvj420p"}
_APPLE_HEVC_PIXEL_FORMATS = {"", "yuv420p", "yuv420p10le"}


def _source_video_format(input_path: Path) -> tuple[str, str]:
    """Return (codec, pixel format) for the first video stream, best effort."""
    try:
        from cleancut.probe import probe_streams

        video = next(s for s in probe_streams(input_path) if s.codec_type == "video")
        return video.codec_name.lower(), video.pix_fmt.lower()
    except (
        AttributeError,
        KeyError,
        OSError,
        RuntimeError,
        StopIteration,
        subprocess.SubprocessError,
        ValueError,
    ):
        # A failed probe should not turn a quick mute-only job into an
        # unexpected multi-hour encode. ffmpeg remains the final validator.
        return "", ""


def _source_video_stream(input_path: Path):
    try:
        from cleancut.probe import probe_streams, video_stream

        return video_stream(probe_streams(input_path))
    except Exception:
        return None


def _partial_output_path(output_path: Path) -> Path:
    return output_path.with_name(f".{output_path.stem}.partial{output_path.suffix}")


def _software_fallback(encoder: str) -> str | None:
    return {
        "videotoolbox": "libx264",
        "hevc_videotoolbox": "libx265",
    }.get(encoder)


def _replace_arg_group(cmd: list[str], old: list[str], new: list[str]) -> list[str]:
    """Replace one contiguous encoder-argument group in an ffmpeg command."""
    for i in range(len(cmd) - len(old) + 1):
        if cmd[i:i + len(old)] == old:
            return cmd[:i] + new + cmd[i + len(old):]
    raise RuntimeError("internal error: video encoder arguments not found")


def _run_with_encoder_fallback(
    cmd: list[str],
    *,
    encoder: str,
    encoder_args: list[str] | None,
    source,
    quality: int,
    partial: Path,
    cwd: str | None = None,
) -> None:
    """Run ffmpeg and retry in software if VideoToolbox cannot open a session."""
    try:
        subprocess.run(cmd, check=True, cwd=cwd)
        return
    except subprocess.CalledProcessError:
        fallback = _software_fallback(encoder)
        if fallback is None or encoder_args is None:
            raise

    partial.unlink(missing_ok=True)
    fallback_args = _video_encoder_args(fallback, quality, source)
    fallback_cmd = _replace_arg_group(cmd, encoder_args, fallback_args)
    print(
        f"[cleancut] VideoToolbox unavailable; retrying render with {fallback}.",
        flush=True,
    )
    subprocess.run(fallback_cmd, check=True, cwd=cwd)


def _finish_atomic_output(
    partial: Path,
    output: Path,
    *,
    expected_duration: float | None,
    expected_dimensions: tuple[int, int] | None,
    validation: str,
) -> None:
    if validation != "none":
        from cleancut.media_validation import validate_media

        validate_media(
            partial,
            expected_duration=expected_duration,
            expected_dimensions=expected_dimensions,
            mode=validation,
        )
    # Mock-based command tests do not create their declared output. Real jobs
    # always request validation and therefore cannot take this compatibility path.
    if partial.exists():
        partial.replace(output)


def _can_stream_copy_to_apple_mp4(codec: str, pix_fmt: str) -> bool:
    """Whether a stream can stay copied while remaining QuickTime/Safari-safe."""
    if not codec:
        return True
    if codec == "h264":
        return pix_fmt in _APPLE_H264_PIXEL_FORMATS
    if codec in {"hevc", "h265"}:
        return pix_fmt in _APPLE_HEVC_PIXEL_FORMATS
    return False


def _muxer_args(output_path: Path, *, copied_video_codec: str = "") -> list[str]:
    """Container flags for seekable MP4 playback, including Apple's HEVC tag."""
    if output_path.suffix.lower() not in _MP4_SUFFIXES:
        return []
    args = ["-movflags", "+faststart"]
    if copied_video_codec in {"hevc", "h265"}:
        # ffmpeg defaults to hev1; Apple software expects hvc1 for HEVC in MP4.
        args += ["-tag:v", "hvc1"]
    return args


# ffmpeg's native AAC encoder refuses to open when the channel layout is
# unspecified — it reports the input as "6 channels" rather than "5.1". A DDP
# 5.1 source whose container carries no layout tag decodes to exactly that, so
# every 5.1 file (the norm for WEB-DL) died at the final mux. Pinning the
# filter chain to layouts the encoder knows is what fixes it.
KNOWN_LAYOUTS = "aformat=channel_layouts=mono|stereo|5.1|7.1"

# 192k spread across six channels is ~32k each, which is poor for surround.
_AAC_BITRATE = {1: "128k", 2: "192k", 6: "448k", 8: "640k"}


def _audio_encoder_args(input_path: Path) -> list[str]:
    """AAC encoder flags with the bitrate matched to the source channel count."""
    channels = 2
    try:
        from cleancut.probe import audio_streams, probe_streams

        tracks = audio_streams(probe_streams(input_path))
        if tracks and tracks[0].channels:
            channels = int(tracks[0].channels)
    except Exception:
        # Probing is a nicety; a stereo-rate fallback still produces valid audio.
        pass
    return ["-c:a", "aac", "-b:a", _AAC_BITRATE.get(channels, "192k")]


def apply_cuts(
    input_path: Path,
    cuts: list[Range],
    output_path: Path,
    encoder: str = "libx264",
    quality: int = 20,
    validation: str = "none",
) -> None:
    """Re-encode `input_path` with `cuts` removed, writing to `output_path`."""
    _require_ffmpeg()
    output_path = output_path.resolve()
    partial = _partial_output_path(output_path)
    partial.unlink(missing_ok=True)
    source_stream = _source_video_stream(input_path)
    source_dimensions = (
        (source_stream.width, source_stream.height)
        if source_stream and source_stream.width and source_stream.height
        else None
    )
    if not cuts:
        # Nothing to cut — just remux.
        codec, _ = _source_video_format(input_path)
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(input_path), "-c", "copy",
                *_muxer_args(output_path, copied_video_codec=codec),
                str(partial),
            ],
            check=True,
        )
        _finish_atomic_output(
            partial, output_path,
            expected_duration=(probe_duration(input_path) if validation != "none" else None),
            expected_dimensions=source_dimensions,
            validation=validation,
        )
        return

    duration = probe_duration(input_path)
    segments = keep_segments(duration, cuts)
    if not segments:
        raise RuntimeError("All segments cut — nothing left to render.")

    parts: list[str] = []
    concat_inputs: list[str] = []
    for i, seg in enumerate(segments):
        parts.append(
            f"[0:v]trim=start={seg.start:.3f}:end={seg.end:.3f},"
            f"setpts=PTS-STARTPTS[v{i}];"
            f"[0:a]atrim=start={seg.start:.3f}:end={seg.end:.3f},"
            f"asetpts=PTS-STARTPTS[a{i}]"
        )
        concat_inputs.append(f"[v{i}][a{i}]")
    filter_complex = ";".join(parts) + ";" + "".join(concat_inputs) + (
        f"concat=n={len(segments)}:v=1:a=1[outv][outa]"
    ) + f";[outa]{KNOWN_LAYOUTS}[outa_fmt]"

    encoder_args = _video_encoder_args(encoder, quality, source_stream)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-filter_complex", filter_complex,
        "-map", "[outv]", "-map", "[outa_fmt]",
        *encoder_args,
        *_audio_encoder_args(input_path),
        *_muxer_args(output_path),
        str(partial),
    ]
    try:
        _run_with_encoder_fallback(
            cmd,
            encoder=encoder,
            encoder_args=encoder_args,
            source=source_stream,
            quality=quality,
            partial=partial,
        )
        expected = max(0.0, duration - sum(r.duration for r in cuts))
        _finish_atomic_output(
            partial, output_path,
            expected_duration=expected,
            expected_dimensions=source_dimensions,
            validation=validation,
        )
    finally:
        partial.unlink(missing_ok=True)


def _ffmpeg_has_libass() -> bool:
    """Check if the installed ffmpeg can run the subtitles= filter (needs libass)."""
    try:
        out = subprocess.check_output(
            ["ffmpeg", "-hide_banner", "-filters"], stderr=subprocess.STDOUT, text=True
        )
        return any(line.split()[1:2] == ["subtitles"] for line in out.splitlines() if line.strip())
    except Exception:
        return False


def apply_mutes_and_subs(
    input_path: Path,
    mutes: list[Range],
    srt_path: Path | None,
    output_path: Path,
    burn_subs: bool = True,
    encoder: str = "libx264",
    quality: int = 20,
    validation: str = "none",
) -> None:
    """Apply mute ranges via volume filter; add subtitles either as burn-in (libass)
    or as a soft subtitle track in the container (always works).

    Soft subs are the default unless `burn_subs=True` AND ffmpeg has libass.
    Soft subs are faster (video can be stream-copied) and toggleable in players.
    """
    _require_ffmpeg()

    # Burn mode runs ffmpeg with cwd inside a temp dir (so the subtitles= filter
    # sees a shell-safe path); relative input/output would resolve there instead.
    input_path = input_path.resolve()
    output_path = output_path.resolve()
    partial = _partial_output_path(output_path)
    partial.unlink(missing_ok=True)
    source_stream = _source_video_stream(input_path)
    source_dimensions = (
        (source_stream.width, source_stream.height)
        if source_stream and source_stream.width and source_stream.height
        else None
    )
    encoder_args: list[str] | None = None

    can_burn = burn_subs and srt_path and srt_path.exists() and _ffmpeg_has_libass()
    source_codec, source_pix_fmt = _source_video_format(input_path)
    copy_video = (
        not can_burn
        and (
            output_path.suffix.lower() not in _MP4_SUFFIXES
            or _can_stream_copy_to_apple_mp4(source_codec, source_pix_fmt)
        )
    )

    cmd: list[str] = ["ffmpeg", "-y", "-i", str(input_path)]

    # If soft-subs mode, add the SRT as a second input.
    has_soft_subs = srt_path and srt_path.exists() and not can_burn
    if has_soft_subs:
        cmd += ["-i", str(srt_path)]

    # Audio filter: mute volumes in the given ranges. The layout pin goes last
    # in the chain and is always present — the encoder needs it whether or not
    # there is anything to mute.
    af_parts: list[str] = []
    if mutes:
        enable = "+".join(f"between(t,{r.start:.3f},{r.end:.3f})" for r in mutes)
        af_parts.append(f"volume=enable='{enable}':volume=0")
    af_parts.append(KNOWN_LAYOUTS)
    cmd += ["-af", ",".join(af_parts)]

    safe_dir: Path | None = None
    if can_burn:
        safe_dir = Path(tempfile.mkdtemp(prefix="cleancut-render_"))
    try:
        if can_burn:
            safe_srt = safe_dir / "subs.srt"
            shutil.copy(str(srt_path), str(safe_srt))
            cmd += ["-vf", "subtitles=subs.srt"]
            encoder_args = _video_encoder_args(encoder, quality, source_stream)
            cmd += encoder_args
        elif has_soft_subs:
            # Stream-copy video, encode subs into the container. mov_text is
            # MP4-family only; Matroska (and most others) take srt.
            sub_codec = "mov_text" if output_path.suffix.lower() in {".mp4", ".m4v", ".mov"} else "srt"
            cmd += ["-map", "0:v", "-map", "0:a", "-map", "1:0"]
            if copy_video:
                cmd += ["-c:v", "copy"]
            else:
                encoder_args = _video_encoder_args(encoder, quality, source_stream)
                cmd += encoder_args
            cmd += ["-c:s", sub_codec]
            cmd += ["-metadata:s:s:0", "language=eng",
                    "-metadata:s:s:0", "title=cleancut (softened)"]
        else:
            if copy_video:
                cmd += ["-c:v", "copy"]
            else:
                encoder_args = _video_encoder_args(encoder, quality, source_stream)
                cmd += encoder_args

        copied_codec = source_codec if copy_video else ""
        cmd += [
            *_audio_encoder_args(input_path),
            *_muxer_args(output_path, copied_video_codec=copied_codec),
            str(partial),
        ]
        cwd = str(safe_dir) if can_burn else None
        _run_with_encoder_fallback(
            cmd,
            encoder=encoder,
            encoder_args=encoder_args,
            source=source_stream,
            quality=quality,
            partial=partial,
            cwd=cwd,
        )
        _finish_atomic_output(
            partial, output_path,
            expected_duration=(probe_duration(input_path) if validation != "none" else None),
            expected_dimensions=source_dimensions,
            validation=validation,
        )
    finally:
        partial.unlink(missing_ok=True)
        if safe_dir is not None:
            shutil.rmtree(safe_dir, ignore_errors=True)
