"""Local/LAN voice replacement. No model dependency in the video process.

Only precise profanity words are eligible. Never synthesize a whole subtitle
or a broad LLM mute. Every failure leaves the existing mute in place.
"""
from __future__ import annotations

import array
import base64
import hashlib
import io
import ipaddress
import json
import math
import os
import socket
import subprocess
import urllib.error
import urllib.request
import wave
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from cleancut.config import Config
from cleancut.editor_ranges import Range, normalize_cuts, shift_after_cuts
from cleancut.edl import EditDecisionList
from cleancut.subtitles import Subtitle

DEFAULT_MODEL = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit"
MAX_WAV_BYTES = 4 * 1024 * 1024


@dataclass
class SpeechClip:
    start: float
    end: float
    path: Path


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("Speech service redirects are not allowed")


def local_service_url(host: str) -> str:
    """Reject cloud destinations, embedded credentials, proxies and redirects."""
    parsed = urlsplit(host.strip())
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.path not in {"", "/"}):
        raise ValueError("Use a local speech host such as http://192.168.1.20:8765")
    nets = [ipaddress.ip_network(n) for n in (
        "127.0.0.0/8", "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16",
        "169.254.0.0/16", "::1/128", "fc00::/7", "fe80::/10",
    )]
    addresses = socket.getaddrinfo(parsed.hostname, parsed.port or 80, type=socket.SOCK_STREAM)
    if not addresses or any(
        not any(ipaddress.ip_address(item[4][0]) in net for net in nets)
        for item in addresses
    ):
        raise ValueError("Speech must stay on localhost or your private LAN")
    return host.strip().rstrip("/")


def _request(config: Config, endpoint: str, payload: dict | None = None,
             *, timeout: float | None = None) -> bytes:
    host = local_service_url(config.speech_host)
    headers = {"Accept": "audio/wav, application/json"}
    token = config.speech_token or os.environ.get("CLEANCUT_SPEECH_TOKEN", "")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    body = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        body = json.dumps(payload).encode()
    req = urllib.request.Request(host + endpoint, data=body, headers=headers)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
    with opener.open(req, timeout=timeout or config.speech_timeout) as response:
        data = response.read(MAX_WAV_BYTES + 1)
    if len(data) > MAX_WAV_BYTES:
        raise ValueError("Speech response exceeds the short-clip limit")
    return data


def check_service(config: Config) -> dict:
    health = json.loads(_request(config, "/health", timeout=5))
    if health.get("service") != "cleancut-speech" or not health.get("voice_cloning"):
        raise ValueError("Not a CleanCut voice service; Ollama's port cannot serve this model")
    if health.get("model") != config.speech_model:
        raise ValueError("The hosted speech model differs from the configured model")
    if not health.get("model_available"):
        raise ValueError("Speech weights are missing; run macos/install-speech.sh on the AI Mac")
    return health


def _reference(subs: list[Subtitle], start: float, end: float) -> Subtitle:
    candidates = [s for s in subs if s.start <= start and s.end >= end
                  and 3 <= s.end - s.start <= 12 and s.text.strip()
                  and not any(line.lstrip().startswith(("-", "–", "—"))
                              for line in s.text.splitlines())]
    if not candidates:
        raise ValueError("No 3–12 second single-speaker reference; keeping the word muted")
    # This is a nearby-utterance heuristic, not speaker diarization. Preview it.
    return min(candidates, key=lambda s: s.end - s.start)


def _run_ffmpeg(*args: str) -> None:
    result = subprocess.run(["ffmpeg", "-nostdin", "-y", "-v", "error", *args],
                            capture_output=True, timeout=60, check=False)
    if result.returncode:
        raise RuntimeError("Speech audio preparation failed: "
                           + result.stderr.decode(errors="replace")[-400:])


def _wav_stats(path: Path) -> tuple[float, float]:
    with wave.open(str(path), "rb") as wav:
        if wav.getsampwidth() != 2 or wav.getnchannels() != 1 or wav.getframerate() != 24000:
            raise ValueError("Expected 24 kHz mono PCM speech")
        samples = array.array("h", wav.readframes(wav.getnframes()))
        if not samples:
            raise ValueError("Empty generated speech")
        rms = math.sqrt(sum(float(s) ** 2 for s in samples) / len(samples)) / 32768
        return wav.getnframes() / wav.getframerate(), rms


def _fit_clip(raw: Path, output: Path, target: float, reference_rms: float | None = None) -> None:
    trimmed = output.with_suffix(".trim.wav")
    try:
        trim = "silenceremove=start_periods=1:start_duration=0.015:start_threshold=-45dB"
        _run_ffmpeg("-i", str(raw), "-af", f"{trim},areverse,{trim},areverse",
                    "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", str(trimmed))
        duration, rms = _wav_stats(trimmed)
        ratio = duration / target
        if rms < 0.002 or not 0.5 <= ratio <= 2.0:
            raise ValueError("Generated word is silent or needs excessive time stretching")
        gain = 1.0 if reference_rms is None else max(0.25, min(2.0, reference_rms / rms))
        fade = min(0.015, target / 5)
        filters = (f"atempo={ratio:.8f},volume={gain:.8f},apad,atrim=duration={target:.8f},"
                   f"afade=t=in:d={fade:.8f},"
                   f"afade=t=out:st={target - fade:.8f}:d={fade:.8f}")
        _run_ffmpeg("-i", str(trimmed), "-af", filters, "-ac", "1", "-ar", "24000",
                    "-c:a", "pcm_s16le", str(output))
        fitted, _ = _wav_stats(output)
        if abs(fitted - target) > 0.005:
            raise ValueError("Generated word did not fit its original interval")
    finally:
        trimmed.unlink(missing_ok=True)


def eligible_words(edl: EditDecisionList, cuts: list[Range]) -> list[dict]:
    words: list[dict] = []
    for d in edl.by_action("mute"):
        for edit in d.word_edits:
            if edit.get("category") != "profanity":
                continue
            try:
                start, end = float(edit["start"]), float(edit["end"])
                before, after = str(edit["text_before"]).strip(), str(edit["text_after"]).strip()
            except (KeyError, TypeError, ValueError):
                continue
            if (not math.isfinite(start + end) or start < d.start or end > d.end
                    or not 0.08 <= end - start <= 3 or not before or not after
                    or before.casefold() == after.casefold() or len(after.split()) > 4
                    or any(start < c.end and end > c.start for c in cuts)):
                continue
            if any(start < w["end"] and end > w["start"] for w in words):
                continue
            words.append({**edit, "start": start, "end": end})
    return words


def prepare_replacements(video: Path, edl: EditDecisionList, subs: list[Subtitle],
                         config: Config, work: Path, *, audio_index: int = 0,
                         cache_dir: Path | None = None,
                         cuts: list[Range] | None = None) -> list[SpeechClip]:
    if config.profanity_audio != "replace":
        return []
    cuts = normalize_cuts(cuts if cuts is not None else [
        Range(d.start, d.end) for d in edl.by_action("cut")
    ])
    edits = eligible_words(edl, cuts)
    if not edits:
        print("[cleancut] Voice replacement: no eligible word timings; using mutes. Rescan old jobs.",
              flush=True)
        return []
    try:
        check_service(config)
    except Exception as exc:  # noqa: BLE001 -- optional speech must never cancel a safe mute.
        print(f"[cleancut] Voice replacement unavailable: {exc}. Keeping word mutes.", flush=True)
        return []
    work.mkdir(parents=True, exist_ok=True)
    clips: list[SpeechClip] = []
    for i, edit in enumerate(edits, 1):
        print(f"[cleancut] Generating replacement {i}/{len(edits)} at {edit['start']:.2f}s",
              flush=True)
        ref_path, raw = work / "reference.wav", work / "generated.wav"
        try:
            ref = _reference(subs, edit["start"], edit["end"])
            stat = video.stat()
            key = hashlib.sha256(json.dumps([
                str(video.resolve()), stat.st_size, stat.st_mtime_ns, audio_index,
                edit, ref.start, ref.end, ref.text, config.speech_host, config.speech_model,
                "word-fit-v1",
            ], sort_keys=True).encode()).hexdigest()
            directory = cache_dir or work
            directory.mkdir(parents=True, exist_ok=True)
            output = directory / f"{key}.wav"
            target = edit["end"] - edit["start"]
            if output.exists():
                try:
                    cached_duration, rms = _wav_stats(output)
                    valid = abs(cached_duration - target) <= 0.005 and rms >= 0.002
                except (OSError, ValueError, wave.Error, EOFError):
                    valid = False
                if not valid:
                    output.unlink()
            if not output.exists():
                _run_ffmpeg("-ss", str(ref.start), "-i", str(video), "-t", str(ref.end-ref.start),
                            "-map", f"0:{audio_index}", "-vn", "-ac", "1", "-ar", "24000",
                            "-c:a", "pcm_s16le", str(ref_path))
                payload = {"model": config.speech_model, "input": edit["text_after"],
                           "ref_text": ref.text, "ref_audio_b64": base64.b64encode(
                               ref_path.read_bytes()).decode(), "response_format": "wav"}
                raw.write_bytes(_request(config, "/v1/audio/speech", payload))
                # Reject JSON/HTML masquerading as a successful response before FFmpeg.
                with wave.open(io.BytesIO(raw.read_bytes()), "rb") as wav:
                    if wav.getnframes() / wav.getframerate() > 15:
                        raise ValueError("Speech response is longer than a replacement word")
                partial = work / f"{key}.partial.wav"
                _, reference_rms = _wav_stats(ref_path)
                _fit_clip(raw, partial, target, reference_rms)
                partial.replace(output)
            shifted = shift_after_cuts(edit["start"], cuts)
            if shifted is not None:
                clips.append(SpeechClip(shifted, shifted + target, output))
        except Exception as exc:  # noqa: BLE001 -- preserve the censor on transport/audio errors.
            print(f"[cleancut] Replacement at {edit['start']:.2f}s skipped: {exc}; word stays muted.",
                  flush=True)
            # Don't wait another two minutes for EVERY word when a host goes
            # offline mid-film. Already prepared clips are still usable.
            if isinstance(exc, (urllib.error.URLError, TimeoutError, ConnectionError)):
                print("[cleancut] Speech service interrupted; remaining words stay muted.", flush=True)
                break
        finally:
            ref_path.unlink(missing_ok=True)
            raw.unlink(missing_ok=True)
    print(f"[cleancut] Voice replacement: {len(clips)}/{len(edits)} words ready; others stay muted.",
          flush=True)
    return clips
