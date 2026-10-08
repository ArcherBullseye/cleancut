"""Isolated, offline Demucs + local speech-leak guard. Never imported by render.

Demucs was trained for music vocals, not cinematic dialogue. Its background is
an estimate, not a guarantee: recognizable speech near an edited word rejects
that background. Whisper can miss faint words; listening review is essential.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import wave
from pathlib import Path
from urllib.request import urlopen

from cleancut.background import MAX_WINDOW, MODEL_FILE, MODEL_URL, SAMPLE_RATE


def install(models: Path) -> None:
    models.mkdir(parents=True, exist_ok=True)
    target = models / MODEL_FILE
    if not target.exists() or not hashlib.sha256(target.read_bytes()).hexdigest().startswith("8726e21a"):
        temporary = target.with_suffix(".partial")
        try:
            digest = hashlib.sha256()
            with urlopen(MODEL_URL, timeout=60) as response, temporary.open("wb") as output:
                while data := response.read(1024 * 1024):
                    digest.update(data)
                    output.write(data)
            if not digest.hexdigest().startswith("8726e21a"):
                raise ValueError("Demucs model checksum mismatch")
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    import whisper

    whisper.load_model("tiny", device="cpu", download_root=str(models))
    print("Separation and speech-verification weights installed locally.", flush=True)


def speech_detected(result: dict, start: float, end: float) -> bool:
    """Reject ANY recognized word/fragment near the censor, not just profanity.

    Adjacent words/music hallucinations can conservatively cause full muting.
    Errors/malformed timestamps are rejected, never assumed clean.
    """
    import math

    if not isinstance(result, dict) or not isinstance(result.get("segments"), list):
        raise TypeError("Missing speech-verification result")
    for segment in result["segments"]:
        words = segment.get("words")
        if not words:
            # Transcription was requested with word timestamps; no timings
            # alongside nonempty text means verification cannot be trusted.
            if segment.get("text", "").strip():
                return True
            continue
        for word in words:
            s, e = float(word["start"]), float(word["end"])
            probability = float(word.get("probability", 1))
            if not math.isfinite(s + e + probability) or e < s:
                raise ValueError("Invalid speech-verification timestamps")
            # Faint leaked phonemes may receive low ASR confidence.
            # No confidence floor: uncertain speech is not evidence of silence.
            # Allow for approximate verification timestamps at either boundary.
            if (word.get("word", "").strip()
                    and s < end + .15 and e > start - .15):
                return True
    return False


def separate_background(model, mix, *, apply=None):
    """Preserve original channel order, scale, and sample count.

    Stereo is processed as stereo. Surround channels are processed as isolated
    mono channels (duplicated to stereo), never silently downmixed. Sum only
    estimated non-vocal stems. Original-minus-vocals reintroduces everything
    the separator failed to estimate, including partially removed dialogue.
    Using stems may lose some effects, and still needs speech verification.
    """
    import numpy as np
    import torch
    if apply is None:
        from demucs.apply import apply_model

        apply = apply_model
    if set(model.sources) != {"drums", "bass", "other", "vocals"}:
        raise ValueError("Unsupported separator source layout")
    background_indices = [model.sources.index(name) for name in ("drums", "bass", "other")]
    background = np.zeros_like(mix)
    groups = [list(range(2))] if mix.shape[1] == 2 else [[i] for i in range(mix.shape[1])]
    torch.manual_seed(0)
    for group in groups:
        pair = mix[:, group].T.copy()
        if len(group) == 1:
            pair = np.repeat(pair, 2, axis=0)
        tensor = torch.from_numpy(pair)
        ref = tensor.mean(0)
        mean, std = ref.mean(), ref.std()
        if float(std) < 1e-6:
            continue
        normalized = (tensor - mean) / std
        # CPU is deliberately conservative: Demucs' complex STFT is not
        # supported consistently by Metal across PyTorch/macOS combinations.
        with torch.inference_mode():
            estimate = apply(model, normalized[None], device="cpu", shifts=0,
                             split=True, overlap=.25, segment=6, progress=False)[0]
        if tuple(estimate.shape) != (len(model.sources), *pair.shape):
            raise ValueError("Separator changed source or sample count")
        # Match Demucs' per-stem denormalization; don't normalize the final bed
        # or add a mixture-consistency residual back from the original audio.
        bed = (estimate[background_indices] * std + mean).sum(dim=0).cpu().numpy()
        if not np.isfinite(bed).all():
            raise ValueError("Separator changed sample count or returned invalid audio")
        background[:, group] = bed.T if len(group) == 2 else bed.mean(axis=0)[:, None]
    if not np.isfinite(background).all() or np.max(np.abs(background)) > 1.5:
        raise ValueError("Unstable background estimate")
    return background


def read_window(path: Path):
    import numpy as np

    with wave.open(str(path), "rb") as wav:
        channels, frames = wav.getnchannels(), wav.getnframes()
        if (wav.getframerate() != SAMPLE_RATE or wav.getsampwidth() != 2
                or channels not in {1, 2, 6, 8} or not 0 < frames / SAMPLE_RATE <= MAX_WINDOW + .05):
            raise ValueError("Expected a bounded PCM window with supported channel count")
        data = wav.readframes(frames)
        if len(data) != frames * channels * 2:
            raise ValueError("Truncated separation input")
    return np.frombuffer(data, dtype="<i2").reshape(-1, channels).astype(np.float32) / 32768


def write_clip(path: Path, background, start: float, end: float) -> None:
    import math

    import numpy as np

    if not math.isfinite(start + end) or not 0 <= start < end <= len(background) / SAMPLE_RATE:
        raise ValueError("Invalid background clip interval")
    first, last = round(start * SAMPLE_RATE), round(end * SAMPLE_RATE)
    clip = background[first:last].copy()
    if len(clip) != last - first or not len(clip):
        raise ValueError("Background does not cover the complete mute")
    fade = min(round(.005 * SAMPLE_RATE), len(clip) // 4)
    if fade:
        clip[:fade] *= np.linspace(0, 1, fade)[:, None]
        clip[-fade:] *= np.linspace(1, 0, fade)[:, None]
    # Reject rather than normalize/clip a hot stem and change the movie's mix.
    if np.max(np.abs(clip)) >= 1:
        raise ValueError("Background would clip")
    partial = path.with_name(f".{path.stem}.{os.getpid()}.partial.wav")
    try:
        with wave.open(str(partial), "wb") as wav:
            wav.setnchannels(clip.shape[1])
            wav.setsampwidth(2)
            wav.setframerate(SAMPLE_RATE)
            wav.writeframes((clip * 32767).astype("<i2").tobytes())
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)


def run(manifest: dict, model, verifier) -> None:
    import numpy as np
    import torch
    from torchaudio.functional import resample

    language = {"eng": "en", "fra": "fr", "spa": "es", "deu": "de", "ita": "it"}.get(
        manifest.get("language"), manifest.get("language"))
    from whisper.tokenizer import LANGUAGES

    if language not in LANGUAGES:
        language = None
    tasks = manifest["tasks"]
    for i, task in enumerate(tasks, 1):
        print(f"[cleancut] Separating background window {i}/{len(tasks)}", flush=True)
        try:
            background = separate_background(model, read_window(Path(task["source"])))
            # Check each channel independently so phase cancellation or a
            # center-only voice cannot hide dialogue in a stereo downmix.
            transcripts = []
            for channel in range(background.shape[1]):
                signal = background[:, channel]
                if float(np.sqrt(np.mean(signal**2))) < .0005:
                    transcripts.append({"segments": []})
                    continue
                audio = resample(torch.from_numpy(signal.copy()), SAMPLE_RATE, 16000).numpy()
                transcripts.append(verifier.transcribe(audio, language=language, fp16=False,
                    word_timestamps=True, condition_on_previous_text=False, temperature=0,
                    verbose=None))
            for target in task["targets"]:
                start, end = float(target["start"]), float(target["end"])
                if any(speech_detected(t, start, end) for t in transcripts):
                    print("[cleancut] Background rejected: possible speech near muted word; full mute retained.",
                          flush=True)
                    continue
                write_clip(Path(target["output"]), background, start, end)
        except Exception as exc:  # noqa: BLE001 -- reject this window, keep other successful words.
            print(f"[cleancut] Background window skipped: {exc}; full mutes retained.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--install", type=Path)
    parser.add_argument("--check", type=Path)
    parser.add_argument("--models", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    if args.install:
        install(args.install)
        return
    import torch
    import whisper
    from demucs.pretrained import get_model

    models = args.check or args.models
    if not models or not (models / MODEL_FILE).is_file() or not (models / "tiny.pt").is_file():
        raise SystemExit("Missing local separation weights; run macos/install-separation.sh")
    if args.check:
        print("Local separation runtime installed.")
        return
    # One worker at a time, including concurrent UI previews. Do not let two
    # copies of separation+ASR compete for an 18 GB Mac's memory.
    with (models / ".worker.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Separation worker is busy; retry when the current preview/render finishes.") from None
        # Explicit file/local repository: NEVER allow inference-time downloads.
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
        model = get_model("955717e8", repo=models)
        verifier = whisper.load_model(str(models / "tiny.pt"), device="cpu")
        run(json.loads(args.manifest.read_text()), model, verifier)


if __name__ == "__main__":
    main()
