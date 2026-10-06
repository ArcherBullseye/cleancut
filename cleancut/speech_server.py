"""Small MLX voice-cloning companion; run on the Apple Silicon AI host.

Accepts bounded WAV bytes, never remote URLs or client-supplied file paths.
Weights must already be downloaded; inference runs offline and serially.
"""
from __future__ import annotations

import base64
import hmac
import io
import os
import platform
import tempfile
import threading
import wave
from pathlib import Path

from flask import Flask, Response, jsonify, request

from cleancut.speech import DEFAULT_MODEL, MAX_WAV_BYTES


def _cached_model(model_id: str) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(model_id, local_files_only=True)


def _generate(model, reference: Path, ref_text: str, text: str) -> bytes:
    import numpy as np
    from mlx_audio.tts.generate import load_audio

    audio = load_audio(str(reference), sample_rate=model.sample_rate)
    pieces = []
    sample_rate = model.sample_rate
    samples = 0
    for result in model.generate(text=text, ref_audio=audio, ref_text=ref_text,
                                 lang_code="English", max_tokens=512, verbose=False):
        sample_rate = result.sample_rate
        piece = np.asarray(result.audio, dtype=np.float32).reshape(-1)
        samples += len(piece)
        if samples > sample_rate * 15 or not np.isfinite(piece).all():
            raise ValueError("Generated audio exceeds the word limit or contains invalid samples")
        pieces.append(piece)
    if not pieces or samples == 0:
        raise ValueError("No speech generated")
    pcm = (np.clip(np.concatenate(pieces), -1, 1) * 32767).astype("<i2").tobytes()
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return buffer.getvalue()


def create_app(*, model_id: str = DEFAULT_MODEL, token: str = "", loader=None,
               generator=None, availability=None) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_WAV_BYTES
    lock = threading.Lock()
    resident = []

    def available():
        if availability is not None:
            return availability()
        try:
            _cached_model(model_id)
            return True
        except Exception:  # noqa: BLE001 -- health reports an absent/incomplete local cache.
            return False

    @app.before_request
    def authenticate():
        if token and not hmac.compare_digest(
            request.headers.get("Authorization", "").encode(), f"Bearer {token}".encode()
        ):
            return jsonify(error="Speech service token is missing or incorrect"), 401

    @app.get("/health")
    def health():
        return jsonify(service="cleancut-speech", voice_cloning=True, model=model_id,
                       model_available=available(), model_loaded=bool(resident))

    @app.post("/v1/audio/speech")
    def speech():
        body = request.get_json(silent=True)
        if not isinstance(body, dict) or body.get("model") != model_id:
            return jsonify(error="Use the model reported by /health"), 400
        text, ref_text = body.get("input"), body.get("ref_text")
        if (not isinstance(text, str) or not text.strip() or len(text) > 100
                or len(text.split()) > 4 or not isinstance(ref_text, str)
                or not ref_text.strip() or len(ref_text) > 1500
                or body.get("response_format", "wav") != "wav"):
            return jsonify(error="Expected a replacement word and reference transcript"), 400
        try:
            encoded = body.get("ref_audio_b64")
            if not isinstance(encoded, str) or len(encoded) > MAX_WAV_BYTES:
                raise ValueError("Expected an uploaded WAV reference, not a file path")
            data = base64.b64decode(encoded, validate=True)
            with wave.open(io.BytesIO(data), "rb") as wav:
                if (wav.getsampwidth() != 2 or wav.getnchannels() != 1
                        or wav.getframerate() != 24000
                        or not 3 <= wav.getnframes() / wav.getframerate() <= 12.05):
                    raise ValueError("Reference must be 3–12 seconds of mono 24 kHz PCM WAV")
                if len(wav.readframes(wav.getnframes())) != wav.getnframes() * 2:
                    raise ValueError("Truncated reference WAV")
        except (ValueError, wave.Error, EOFError) as exc:
            return jsonify(error=str(exc)), 400
        if not lock.acquire(blocking=False):
            return jsonify(error="Speech service is busy; retry after the current word"), 503
        try:
            if not resident:
                if loader is not None:
                    resident.append(loader(model_id))
                else:
                    from mlx_audio.tts.utils import load_model

                    resident.append(load_model(_cached_model(model_id)))
            with tempfile.TemporaryDirectory(prefix="cleancut-speech-") as directory:
                reference = Path(directory) / "reference.wav"
                reference.write_bytes(data)
                output = (generator or _generate)(resident[0], reference, ref_text, text)
            return Response(output, mimetype="audio/wav", headers={"Cache-Control": "no-store"})
        except Exception as exc:  # noqa: BLE001 -- isolate model failures from the service.
            print(f"[cleancut-speech] Generation failed: {type(exc).__name__}: {exc}", flush=True)
            return jsonify(error="Voice generation failed; CleanCut will keep the word muted"), 503
        finally:
            try:
                if loader is None and resident:
                    import mlx.core as mx

                    mx.clear_cache()  # release temporary Metal allocations, keep model weights.
            finally:
                lock.release()

    return app


def main() -> None:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise SystemExit("The speech companion requires an Apple Silicon Mac")
    host = os.environ.get("CLEANCUT_SPEECH_BIND", "127.0.0.1")
    token = os.environ.get("CLEANCUT_SPEECH_TOKEN", "")
    if host not in {"127.0.0.1", "localhost", "::1"} and not token:
        raise SystemExit("Set CLEANCUT_SPEECH_TOKEN before allowing LAN connections")
    from waitress import serve

    model_id = os.environ.get("CLEANCUT_SPEECH_MODEL", DEFAULT_MODEL)
    port = int(os.environ.get("CLEANCUT_SPEECH_PORT", "8765"))
    print(f"CleanCut speech: http://{host}:{port} · {model_id}", flush=True)
    serve(create_app(model_id=model_id, token=token), host=host,
          port=port, threads=2)


if __name__ == "__main__":
    main()
