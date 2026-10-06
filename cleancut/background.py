"""Short-window, local soundtrack restoration for precise word mutes.

The separator lives in its own process/environment. It never gets a URL or a
movie path: only bounded local audio windows. Missing dependencies, poor stems,
or recognizable speech in a background leave the full-mix mute intact.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import wave
from pathlib import Path

from cleancut.config import Config
from cleancut.editor_ranges import Range, normalize_cuts, shift_after_cuts
from cleancut.edl import EditDecisionList
from cleancut.probe import probe_duration
from cleancut.speech import SpeechClip, _run_ffmpeg

MODEL_FILE = "955717e8-8726e21a.th"
MODEL_URL = "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/" + MODEL_FILE
ALGORITHM = "htdemucs-residual-tiny-speech-guard-v1"
SAMPLE_RATE = 44100
MAX_WINDOW = 16.0


def runtime_python() -> Path:
    return Path(os.environ.get("CLEANCUT_SEPARATION_PYTHON", str(
        Path(__file__).resolve().parents[1] / ".venv-separation/bin/python"
    )))


def model_directory() -> Path:
    default = Path.home() / "Library/Application Support/CleanCut/separation-models"
    if sys.platform != "darwin":
        default = Path.home() / ".cache/cleancut/separation-models"
    data_root = os.environ.get("CLEANCUT_DATA_DIR") or os.environ.get("DATA_DIR")
    if data_root:
        default = Path(data_root) / "separation-models"
    return Path(os.environ.get("CLEANCUT_SEPARATION_MODEL_DIR", str(default)))


def check_runtime() -> dict:
    python, models = runtime_python(), model_directory()
    if not python.is_file() or not (models / MODEL_FILE).is_file() or not (models / "tiny.pt").is_file():
        raise ValueError("Run ./macos/install-separation.sh on the Mac running CleanCut first")
    result = subprocess.run([str(python), "-m", "cleancut.separation_worker", "--check", str(models)],
                            capture_output=True, text=True, timeout=30, check=False, env=worker_env())
    if result.returncode:
        raise ValueError("Separation installation is incomplete; rerun macos/install-separation.sh")
    return {"model": "Demucs htdemucs", "verification": "local Whisper tiny", "ready": True}


def worker_env() -> dict[str, str]:
    env = os.environ.copy()
    root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    return env


def eligible_mutes(edl: EditDecisionList, cuts: list[Range]) -> list[Range]:
    """Only whole word-timed decisions, never broad event/scene mutes.

    Restoring a background inside another censor could undermine that censor,
    so overlapping decisions and anything intersecting a cut are left muted.
    """
    decisions = edl.by_action("mute")
    out = []
    for d in decisions:
        if (not d.word_edits or d.source != "whisper-word"
                or not math.isfinite(d.start + d.end) or d.start < 0
                or not .08 <= d.duration <= 8
                or any(d.start < c.end and d.end > c.start for c in cuts)
                or any(other is not d and d.start < other.end and d.end > other.start
                       for other in decisions)):
            continue
        valid = True
        for word in d.word_edits:
            try:
                start, end = float(word["start"]), float(word["end"])
                valid &= (math.isfinite(start + end) and d.start <= start < end <= d.end
                          and .08 <= end - start <= 3 and bool(word["text_before"].strip())
                          and word["category"] in {"profanity", "drugs", "sex", "violence"})
            except (KeyError, TypeError, ValueError, AttributeError):
                valid = False
        if valid:
            out.append(Range(d.start, d.end))
    return sorted(out, key=lambda r: r.start)


def window_groups(targets: list[Range], duration: float) -> list[tuple[Range, list[Range]]]:
    """Coalesce nearby targets without ever separating a movie-sized buffer."""
    groups: list[tuple[Range, list[Range]]] = []
    for target in targets:
        if target.end > duration:
            continue
        window = Range(max(0, target.start - 3), min(duration, target.end + 3))
        if groups and window.start <= groups[-1][0].end and window.end - groups[-1][0].start <= MAX_WINDOW:
            groups[-1][0].end = max(groups[-1][0].end, window.end)
            groups[-1][1].append(target)
        else:
            groups.append((window, [target]))
    return groups


def valid_clip(path: Path, duration: float, channels: int) -> bool:
    try:
        with wave.open(str(path), "rb") as wav:
            frames = wav.getnframes()
            return (wav.getsampwidth() == 2 and wav.getframerate() == SAMPLE_RATE
                    and wav.getnchannels() == channels
                    and abs(frames / SAMPLE_RATE - duration) < .002
                    and len(wav.readframes(frames)) == frames * channels * 2)
    except (OSError, ValueError, wave.Error, EOFError):
        return False


def prepare_background(video: Path, edl: EditDecisionList, config: Config, work: Path,
                       *, audio_index: int, channels: int, cuts: list[Range],
                       cache_dir: Path | None = None, language: str = "eng",
                       only: Range | None = None) -> list[SpeechClip]:
    if not config.preserve_background:
        return []
    cuts = normalize_cuts(cuts)
    targets = eligible_mutes(edl, cuts)
    if only is not None:
        targets = [t for t in targets if t == only]
    if not targets:
        print("[cleancut] Background preservation: no eligible word mutes; rescan older jobs.", flush=True)
        return []
    if channels not in {1, 2, 6, 8}:
        print("[cleancut] Background preservation: unsupported channel layout; keeping full mutes.", flush=True)
        return []
    work.mkdir(parents=True, exist_ok=True)
    directory = cache_dir or work / "cache"
    directory.mkdir(parents=True, exist_ok=True)
    stat = video.stat()
    cache = {}
    for target in targets:
        key = hashlib.sha256(json.dumps([
            str(video.resolve()), stat.st_size, stat.st_mtime_ns, audio_index, channels,
            target.start, target.end, language, ALGORITHM,
        ]).encode()).hexdigest()
        cache[(target.start, target.end)] = directory / f"{key}.wav"
    pending = [t for t in targets if not valid_clip(cache[(t.start, t.end)], t.duration, channels)]
    if pending:
        try:
            check_runtime()
            tasks = []
            for i, (window, group) in enumerate(window_groups(pending, probe_duration(video))):
                source = work / f"window-{i}.wav"
                _run_ffmpeg("-ss", str(window.start), "-i", str(video), "-t", str(window.duration),
                            "-map", f"0:{audio_index}", "-vn", "-ar", str(SAMPLE_RATE),
                            "-c:a", "pcm_s16le", str(source))
                tasks.append({"source": str(source.resolve()), "targets": [
                    {"start": t.start - window.start, "end": t.end - window.start,
                     "output": str(cache[(t.start, t.end)].resolve())} for t in group
                ]})
            manifest = work / "manifest.json"
            manifest.write_text(json.dumps({"tasks": tasks, "language": language}))
            print(f"[cleancut] Separating background: {len(tasks)} short window(s), {len(pending)} word mute(s)",
                  flush=True)
            # No remote repo path or auto-download on the inference path. The
            # entire Torch worker exits before we load/generate replacement TTS.
            subprocess.run([
                str(runtime_python()), "-u", "-m", "cleancut.separation_worker",
                "--models", str(model_directory()), "--manifest", str(manifest),
            ], check=True, timeout=max(600, len(tasks) * 300), env=worker_env())
        except Exception as exc:  # noqa: BLE001 -- censorship remains intact on optional failures.
            print(f"[cleancut] Background separation unavailable: {exc}; keeping full word mutes.", flush=True)
    clips = []
    for target in targets:
        path = cache[(target.start, target.end)]
        if valid_clip(path, target.duration, channels):
            start = shift_after_cuts(target.start, cuts)
            if start is not None:
                clips.append(SpeechClip(start, start + target.duration, path))
    print(f"[cleancut] Background preservation: {len(clips)}/{len(targets)} word(s) ready; others stay fully muted.",
          flush=True)
    return clips


def preview_mix(video: Path, target: Range, output: Path, *, audio_index: int,
                overlays: list[SpeechClip]) -> None:
    """Audition source-timed overlays with two seconds of unchanged context."""
    start = max(0, target.start - 2)
    duration = min(probe_duration(video), target.end + 2) - start
    args = ["-ss", str(start), "-i", str(video)]
    graph = [(f"[0:{audio_index}]asetpts=PTS-STARTPTS,"
              f"volume=0:enable='between(t,{target.start-start:.9f},{target.end-start:.9f})'[original]")]
    labels = ["[original]"]
    for i, clip in enumerate(overlays, 1):
        args.extend(["-i", str(clip.path)])
        label = f"overlay{i}"
        graph.append(f"[{i}:a]atrim=duration={clip.end-clip.start:.9f},asetpts=PTS-STARTPTS,"
                     f"adelay={round((clip.start-start)*1000)}:all=1[{label}]")
        labels.append(f"[{label}]")
    graph.append("".join(labels) + f"amix=inputs={len(labels)}:duration=first:normalize=0:"
                 "dropout_transition=0[out]")
    _run_ffmpeg(*args, "-filter_complex", ";".join(graph), "-map", "[out]", "-vn",
                "-t", str(duration), "-c:a", "pcm_s16le", str(output))
