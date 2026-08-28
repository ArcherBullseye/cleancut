"""Whisper-based transcription with word-level timestamps and MPS acceleration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from cleancut.subtitles import Subtitle


@dataclass
class Word:
    """A single word with its start/end time in seconds and probability."""
    start: float
    end: float
    text: str
    probability: float = 1.0


def _autodetect_device(word_timestamps: bool = False) -> str:
    """Prefer MPS (Apple Silicon GPU) when available, else CUDA, else CPU.

    Important caveat: Whisper's word-timestamp DTW alignment uses float64 ops
    that MPS doesn't support, and PYTORCH_ENABLE_MPS_FALLBACK does not cover
    those in-place dtype conversions. When word timestamps are requested we
    fall back to CPU automatically so the run actually completes.
    """
    try:
        import torch  # type: ignore
        if torch.backends.mps.is_available():
            if word_timestamps:
                return "cpu"
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def _resolve_device(requested: str | None, word_timestamps: bool) -> str:
    """Resolve the actual device to use, with a notice if we override."""
    if requested is None:
        return _autodetect_device(word_timestamps=word_timestamps)
    if requested == "mps" and word_timestamps:
        print(
            "[cleancut] WARNING: MPS does not support Whisper's word-timestamp "
            "float64 ops — falling back to CPU. Disable word timestamps with "
            "--no-word-timestamps to use MPS."
        )
        return "cpu"
    return requested


def transcribe(
    video_path: Path,
    model_name: str = "large-v3",
    device: str | None = None,
    word_timestamps: bool = True,
    language: str | None = None,
    audio_path: Path | None = None,
) -> tuple[list[Subtitle], list[Word]]:
    """Run Whisper on `video_path` (or on `audio_path` if provided).

    Returns (segment-level subtitles, word-level words). If `word_timestamps`
    is False, the words list will be empty.

    When `audio_path` is provided (e.g. a pre-extracted English dub track),
    Whisper transcribes that instead — useful for multi-track files where
    Whisper would otherwise default to track 0.
    """
    try:
        import whisper  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "Whisper not installed. Install with: pip install -e '.[whisper]'"
        ) from e

    device = device or _autodetect_device()

    # MPS doesn't support fp16; force fp32.
    fp16 = device == "cuda"

    model = whisper.load_model(model_name, device=device)
    input_for_whisper = str(audio_path) if audio_path else str(video_path)
    result = model.transcribe(
        input_for_whisper,
        verbose=False,
        word_timestamps=word_timestamps,
        fp16=fp16,
        language=language,
    )

    subs: list[Subtitle] = []
    words: list[Word] = []
    for i, seg in enumerate(result.get("segments", []), start=1):
        subs.append(
            Subtitle(
                index=i,
                start=float(seg["start"]),
                end=float(seg["end"]),
                text=str(seg["text"]).strip(),
            )
        )
        if word_timestamps:
            for w in seg.get("words", []) or []:
                # Whisper sometimes emits empty word strings — skip those.
                token = str(w.get("word", "")).strip()
                if not token:
                    continue
                words.append(
                    Word(
                        start=float(w["start"]),
                        end=float(w["end"]),
                        text=token,
                        probability=float(w.get("probability", 1.0)),
                    )
                )

    return subs, words
