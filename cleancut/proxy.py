"""Low-resolution, timestamp-preserving proxy for visual analysis.

The edit decisions remain in the original video's timeline. Only frame-heavy
scene/NudeNet/VLM analysis uses the proxy; audio, subtitles, and final rendering
always use the source file.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
from pathlib import Path

from cleancut import cache
from cleancut.probe import probe_streams, video_stream


def _encoder_available(name: str) -> bool:
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return name in result.stdout


def analysis_source(video: Path, max_height: int = 720) -> Path:
    """Return `video` or a cached analysis proxy when the source is larger.

    Proxy generation is atomic and keyed by source fingerprint through the
    normal cache metadata. A killed encode can therefore never be mistaken for
    a reusable proxy on the next job.
    """
    if max_height <= 0:
        return video
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found on PATH; cannot build analysis proxy")

    source = video_stream(probe_streams(video))
    if source is None or not source.width or not source.height:
        return video
    if source.height <= max_height:
        return video

    target_width = max(2, round(source.width * max_height / source.height / 2) * 2)
    h = cache.config_hash(
        version=1,
        width=target_width,
        height=max_height,
        fps=source.avg_frame_rate,
    )
    cached = cache.load(video, "analysis_proxy", h)
    if cached:
        path = Path(cached.get("path", ""))
        if path.is_file() and path.stat().st_size > 0:
            return path

    cache.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    output = cache.artifact_path(video, "analysis_proxy", h, ".mp4")
    partial = output.with_name(f".{output.stem}.partial{output.suffix}")
    partial.unlink(missing_ok=True)

    use_videotoolbox = (
        platform.system() == "Darwin" and _encoder_available("h264_videotoolbox")
    )
    encoder = (
        [
            "-c:v", "h264_videotoolbox", "-q:v", "38", "-b:v", "0",
            "-allow_sw", "1",
        ]
        if use_videotoolbox
        else ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "30"]
    )
    cmd = [
        "ffmpeg", "-nostdin", "-y", "-v", "error",
        "-i", str(video),
        "-map", "0:v:0", "-an", "-sn", "-dn",
        "-vf", f"scale={target_width}:{max_height}:flags=fast_bilinear,format=yuv420p",
        *encoder,
        "-fps_mode", "vfr",
        "-movflags", "+faststart",
        str(partial),
    ]
    try:
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            if not use_videotoolbox:
                raise
            partial.unlink(missing_ok=True)
            hardware_args = encoder
            software_args = ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "30"]
            for i in range(len(cmd) - len(hardware_args) + 1):
                if cmd[i:i + len(hardware_args)] == hardware_args:
                    cmd = cmd[:i] + software_args + cmd[i + len(hardware_args):]
                    break
            else:
                raise RuntimeError("analysis proxy encoder arguments not found")
            subprocess.run(cmd, check=True)
        if not partial.exists() or partial.stat().st_size == 0:
            raise RuntimeError("ffmpeg produced an empty analysis proxy")
        partial.replace(output)
    finally:
        partial.unlink(missing_ok=True)

    cache.save(video, "analysis_proxy", h, {
        "path": str(output),
        "width": target_width,
        "height": max_height,
    })
    return output
