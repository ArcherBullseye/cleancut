"""ffprobe wrappers: list audio/subtitle tracks, find sidecar subtitles, extract tracks."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


def _mkstemp_path(prefix: str, suffix: str) -> Path:
    """mkstemp that closes the fd immediately — we only want the path."""
    fd, name = tempfile.mkstemp(prefix=prefix, suffix=suffix)
    os.close(fd)
    return Path(name)


# Container-internal subtitle codecs that produce text we can scan.
TEXT_SUBTITLE_CODECS = {"subrip", "srt", "mov_text", "ass", "ssa", "webvtt"}
# Image-based subtitle codecs that would need OCR — we skip these.
IMAGE_SUBTITLE_CODECS = {"dvd_subtitle", "hdmv_pgs_subtitle", "dvb_subtitle"}

# Filename suffix → language preference for sidecar .srt discovery.
# Higher score wins.
LANG_PREFS: dict[str, int] = {
    "": 5,           # bare Movie.srt — usually the primary language
    "en": 10,
    "eng": 10,
    "english": 10,
    "en-us": 9,
    "en-gb": 9,
}


@dataclass
class Stream:
    index: int
    codec_name: str
    codec_type: str  # "video" | "audio" | "subtitle" | "data"
    language: str = "und"
    title: str = ""
    channels: int | None = None


def probe_duration(path: Path) -> float:
    """Container duration in seconds via ffprobe."""
    if not shutil.which("ffprobe"):
        raise RuntimeError("ffprobe not found on PATH. It ships with ffmpeg.")
    out = subprocess.check_output(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "json", str(path),
        ]
    )
    return float(json.loads(out)["format"]["duration"])


def probe_streams(video: Path) -> list[Stream]:
    """Return every stream in the file."""
    if not shutil.which("ffprobe"):
        raise RuntimeError("ffprobe not found on PATH.")
    out = subprocess.check_output(
        [
            "ffprobe", "-v", "error",
            "-show_streams", "-of", "json",
            str(video),
        ]
    )
    data = json.loads(out)
    streams: list[Stream] = []
    for s in data.get("streams", []):
        streams.append(
            Stream(
                index=int(s.get("index", -1)),
                codec_name=str(s.get("codec_name", "")),
                codec_type=str(s.get("codec_type", "")),
                language=str(s.get("tags", {}).get("language", "und")),
                title=str(s.get("tags", {}).get("title", "")),
                channels=s.get("channels"),
            )
        )
    return streams


def audio_streams(streams: list[Stream]) -> list[Stream]:
    return [s for s in streams if s.codec_type == "audio"]


def subtitle_streams(streams: list[Stream]) -> list[Stream]:
    return [s for s in streams if s.codec_type == "subtitle"]


def pick_audio_track(
    streams: list[Stream], requested: int | None, prefer_language: str | None = "eng",
) -> Stream | None:
    """Pick which audio track to feed Whisper.

    Order: explicit request → preferred language → first track.
    `requested` is 0-indexed within the audio streams list (audio:0, audio:1, …).
    """
    audios = audio_streams(streams)
    if not audios:
        return None
    if requested is not None:
        if 0 <= requested < len(audios):
            return audios[requested]
        raise ValueError(f"audio track {requested} not available (have {len(audios)})")
    if prefer_language:
        prefer = {prefer_language, prefer_language[:2]}  # e.g. "eng" and "en"
        for s in audios:
            if s.language in prefer:
                return s
    return audios[0]


def extract_audio_to_wav(video: Path, audio_stream_index: int) -> Path:
    """Extract a specific audio stream to a 16kHz mono WAV (Whisper's preferred format).

    `audio_stream_index` is the absolute stream index from ffprobe, not a track ordinal.
    """
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found on PATH.")
    tmp = _mkstemp_path(prefix="cleancut_", suffix=".wav")
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-i", str(video),
            "-map", f"0:{audio_stream_index}",
            "-ac", "1", "-ar", "16000",
            "-acodec", "pcm_s16le",
            str(tmp),
        ],
        check=True,
    )
    return tmp


def extract_text_subtitle(video: Path, subtitle_stream_index: int) -> Path | None:
    """Extract an embedded text subtitle stream to a temporary .srt file."""
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found on PATH.")
    tmp = _mkstemp_path(prefix="cleancut_", suffix=".srt")
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-v", "error",
                "-i", str(video),
                "-map", f"0:{subtitle_stream_index}",
                "-c:s", "srt",
                str(tmp),
            ],
            check=True,
        )
        return tmp
    except subprocess.CalledProcessError:
        tmp.unlink(missing_ok=True)
        return None


def pick_embedded_subtitle(
    streams: list[Stream], prefer_language: str = "eng",
) -> Stream | None:
    """Pick the best embedded text-based subtitle stream. Skips image subs."""
    text_subs = [s for s in subtitle_streams(streams) if s.codec_name in TEXT_SUBTITLE_CODECS]
    if not text_subs:
        return None
    prefer = {prefer_language, prefer_language[:2]}
    for s in text_subs:
        if s.language in prefer:
            return s
    return text_subs[0]


def _lang_prefs(prefer_language: str) -> dict[str, int]:
    """Score map for sidecar suffixes, boosting the requested language."""
    prefs = dict(LANG_PREFS)
    if prefer_language:
        prefs[prefer_language.lower()] = 12
        prefs[prefer_language[:2].lower()] = 11
    return prefs


def find_sidecar_subtitle(video: Path, prefer_language: str = "eng") -> Path | None:
    """Find the best .srt file next to the video, preferring `prefer_language`.

    Recognizes Plex-style suffixes: Movie.srt, Movie.en.srt, Movie.eng.srt,
    Movie.English.srt. Also looks inside a `Subs/` subfolder if present.
    """
    prefs = _lang_prefs(prefer_language)
    stem = video.stem
    candidates: list[tuple[int, Path]] = []
    search_dirs = [video.parent]
    subs_dir = video.parent / "Subs"
    if subs_dir.is_dir():
        search_dirs.append(subs_dir)

    for d in search_dirs:
        for p in d.glob("*.srt"):
            # Match either the exact stem or stem.suffix patterns.
            name = p.stem
            if name == stem:
                candidates.append((prefs.get("", 5), p))
                continue
            if name.startswith(stem + "."):
                suffix = name[len(stem) + 1:].lower()
                score = prefs.get(suffix, prefs.get(suffix[:2], 1))
                # Penalize SDH/forced flavors.
                if "sdh" in suffix or "forced" in suffix:
                    score -= 3
                candidates.append((score, p))
            elif p.parent == subs_dir:
                # Inside Subs/, names usually don't include the movie name.
                suffix = name.lower()
                score = prefs.get(suffix, prefs.get(suffix[:2], 2))
                if "sdh" in suffix or "forced" in suffix:
                    score -= 3
                candidates.append((score, p))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]
