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


def _video_encoder_args(encoder: str, quality: int) -> list[str]:
    """Return ffmpeg flags for the chosen video encoder."""
    if encoder == "videotoolbox":
        # videotoolbox uses -q:v (higher = better). Map CRF-ish quality to a sensible q.
        # CRF 18 ~ q 54, CRF 20 ~ q 50, CRF 23 ~ q 44.
        q = max(30, min(100, 90 - quality * 2))
        return ["-c:v", "h264_videotoolbox", "-q:v", str(q), "-b:v", "0"]
    # libx264 default
    return ["-c:v", "libx264", "-preset", "slow", "-crf", str(quality)]


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
) -> None:
    """Re-encode `input_path` with `cuts` removed, writing to `output_path`."""
    _require_ffmpeg()
    if not cuts:
        # Nothing to cut — just remux.
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(input_path), "-c", "copy", str(output_path)],
            check=True,
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

    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-filter_complex", filter_complex,
        "-map", "[outv]", "-map", "[outa_fmt]",
        *_video_encoder_args(encoder, quality),
        *_audio_encoder_args(input_path),
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


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

    can_burn = burn_subs and srt_path and srt_path.exists() and _ffmpeg_has_libass()

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
            cmd += _video_encoder_args(encoder, quality)
        elif has_soft_subs:
            # Stream-copy video, encode subs into the container. mov_text is
            # MP4-family only; Matroska (and most others) take srt.
            sub_codec = "mov_text" if output_path.suffix.lower() in {".mp4", ".m4v", ".mov"} else "srt"
            cmd += ["-map", "0:v", "-map", "0:a", "-map", "1:0"]
            cmd += ["-c:v", "copy"]
            cmd += ["-c:s", sub_codec]
            cmd += ["-metadata:s:s:0", "language=eng",
                    "-metadata:s:s:0", "title=cleancut (softened)"]
        else:
            cmd += ["-c:v", "copy"]

        cmd += [*_audio_encoder_args(input_path), str(output_path)]
        cwd = str(safe_dir) if can_burn else None
        subprocess.run(cmd, check=True, cwd=cwd)
    finally:
        if safe_dir is not None:
            shutil.rmtree(safe_dir, ignore_errors=True)
