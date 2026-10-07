from __future__ import annotations

import base64
import io
import json
import math
import shutil
import subprocess
import wave
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

from cleancut.config import Config
from cleancut.editor_ranges import Range
from cleancut.edl import EditDecision, EditDecisionList
from cleancut.speech import (
    DEFAULT_MODEL,
    SpeechClip,
    _fit_clip,
    _reference,
    _wav_stats,
    check_service,
    eligible_words,
    local_service_url,
    prepare_replacements,
)
from cleancut.speech_server import create_app
from cleancut.subtitles import Subtitle, scan_words
from cleancut.transcribe import Word


def wav_bytes(duration=3.0, frequency=700, rate=24000):
    import array

    buffer = io.BytesIO()
    samples = array.array("h", (int(4000 * math.sin(i * 2 * math.pi * frequency / rate))
                               for i in range(round(duration * rate))))
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(samples.tobytes())
    return buffer.getvalue()


def word_decision(start=2.2, end=2.8, **updates):
    d = EditDecision(start=start - 0.05, end=end + 0.05, action="mute", category="profanity",
                     source="whisper-word", word_edits=[{
                         "start": start, "end": end, "text_before": "damn", "text_after": "darn",
                         "category": "profanity",
                     }])
    return replace(d, **updates)


def speech_payload(**updates):
    return {"model": DEFAULT_MODEL, "input": "darn", "ref_text": "That is a damn good idea.",
            "ref_audio_b64": base64.b64encode(wav_bytes()).decode(), **updates}


def fake_speech_app(**updates):
    return create_app(loader=lambda name: object(), generator=lambda *args: wav_bytes(0.6),
                      availability=lambda: True, **updates)


def test_word_metadata_contains_only_offending_word_and_survives_review(tmp_path):
    cfg = Config.load_defaults()
    words = [Word(0, .5, "What"), Word(.5, .8, "the"), Word(.8, 1.2, "fuck"),
             Word(1.2, 1.6, "happened?")]
    scanned = scan_words(words, cfg)
    assert len(scanned.decisions) == 1
    d = scanned.decisions[0]
    assert d.text_before == "fuck"
    assert d.word_edits[0]["start"] == .8
    merged = scanned.pad(.15).merge_overlapping(.5).sorted()
    assert merged.decisions[0].word_edits == d.word_edits
    path = tmp_path / "edl.json"
    merged.to_json(path)
    from webapp import review

    edited = review.load_edl(path)
    review.apply_edit(edited, 0, {"accepted": False})
    review.save_edl(path, edited)
    restored = EditDecisionList.from_json(path)
    assert restored.decisions[0].word_edits == d.word_edits
    assert not restored.decisions[0].accepted


def test_merging_retains_multiple_words_without_mutating_input():
    first, second = word_decision(), word_decision(2.84, 3.2)
    edl = EditDecisionList(decisions=[first, second]).merge_overlapping(.5)
    assert len(edl.decisions[0].word_edits) == 2
    assert len(first.word_edits) == 1


def test_word_mute_padding_and_merging_do_not_silence_surrounding_words():
    first, second = word_decision(1, 1.3), word_decision(1.5, 1.8)
    # Simulate raw scan intervals before the pipeline's padding pass.
    first = replace(first, start=1, end=1.3)
    second = replace(second, start=1.5, end=1.8)
    first.word_edits[0]["next_word_start"] = 1.35  # an unflagged word between these two
    padded = EditDecisionList(decisions=[first, second]).pad(.15).merge_overlapping(.5)
    assert len(padded.decisions) == 2
    assert padded.decisions[0].end < 1.4 < padded.decisions[1].start
    assert abs(padded.decisions[0].start - .96) < .001


@pytest.mark.parametrize("updates", [
    {"accepted": False}, {"action": "cut"}, {"action": "keep"}, {"word_edits": []},
    {"end": 2.5}, {"start": 2.4},
])
def test_only_accepted_precise_mutes_are_eligible(updates):
    assert not eligible_words(EditDecisionList(decisions=[replace(word_decision(), **updates)]), [])


def test_cut_intersections_and_non_profanity_never_synthesize():
    d = word_decision()
    assert not eligible_words(EditDecisionList(decisions=[d]), [Range(2.7, 4)])
    d.word_edits[0]["category"] = "sex"
    assert not eligible_words(EditDecisionList(decisions=[d]), [])


def test_unchanged_or_nonfinite_word_is_not_eligible():
    d = word_decision()
    d.word_edits[0]["text_after"] = "damn"
    assert not eligible_words(EditDecisionList(decisions=[d]), [])
    d.word_edits[0].update(text_after="darn", start=float("nan"))
    assert not eligible_words(EditDecisionList(decisions=[d]), [])


@pytest.mark.parametrize("subs", [
    [], [Subtitle(1, 2, 3, "Too short.")], [Subtitle(1, 0, 14, "Too long.")],
    [Subtitle(1, 0, 4, "- First speaker\n- Second speaker")],
])
def test_reference_rejects_missing_short_long_and_marked_multispeaker_audio(subs):
    with pytest.raises(ValueError, match="reference"):
        _reference(subs, 2.2, 2.8)


def test_private_lan_allowed_and_public_service_rejected():
    with patch("cleancut.speech.socket.getaddrinfo", return_value=[(2, 1, 6, "", ("192.168.1.20", 8765))]):
        assert local_service_url("http://ai-mac.local:8765/") == "http://ai-mac.local:8765"
    with patch("cleancut.speech.socket.getaddrinfo", return_value=[(2, 1, 6, "", ("8.8.8.8", 443))]), \
         pytest.raises(ValueError, match="private LAN"):
        local_service_url("https://public.example")


@pytest.mark.parametrize("url", ["ftp://localhost", "http://a:b@localhost", "http://localhost/path",
                                 "http://localhost?token=x", "http://localhost#fragment"])
def test_speech_host_disallows_credentials_and_non_service_urls(url):
    with pytest.raises(ValueError):
        local_service_url(url)


def test_health_distinguishes_ollama_and_wrong_or_missing_model():
    for health in ({"models": []}, {"service": "cleancut-speech", "voice_cloning": True,
                                   "model": "wrong"},
                   {"service": "cleancut-speech", "voice_cloning": True, "model": DEFAULT_MODEL,
                    "model_available": False}):
        with patch("cleancut.speech._request", return_value=json.dumps(health).encode()), \
             pytest.raises(ValueError):
            check_service(Config())


def test_service_outage_retains_mute_without_running_ffmpeg(tmp_path):
    edl = EditDecisionList(decisions=[word_decision()])
    with patch("cleancut.speech.check_service", side_effect=OSError("offline")), \
         patch("cleancut.speech._run_ffmpeg") as ffmpeg:
        assert prepare_replacements(tmp_path / "video", edl, [],
                                    Config(profanity_audio="replace"), tmp_path) == []
    ffmpeg.assert_not_called()
    assert edl.decisions[0].action == "mute"


def test_companion_accepts_remote_reference_bytes_and_keeps_model_resident():
    calls = []
    app = create_app(loader=lambda name: calls.append(name),
                     generator=lambda *args: wav_bytes(.6), availability=lambda: True)
    client = app.test_client()
    assert client.get("/health").json["model_available"]
    assert not calls  # health is not a model load or a paid/network request.
    for _ in range(2):
        response = client.post("/v1/audio/speech", json=speech_payload())
        assert response.status_code == 200
        assert response.mimetype == "audio/wav"
        with wave.open(io.BytesIO(response.data)) as wav:
            assert wav.getnframes() == 14400
    assert calls == [DEFAULT_MODEL]


def test_companion_authenticates_health_and_inference():
    client = fake_speech_app(token="shared-secret").test_client()
    assert client.get("/health").status_code == 401
    assert client.post("/v1/audio/speech", json=speech_payload()).status_code == 401
    assert client.get("/health", headers={"Authorization": "Bearer shared-secret"}).status_code == 200


@pytest.mark.parametrize("updates", [
    {"model": "other"}, {"ref_audio_b64": "/etc/passwd"}, {"ref_audio_b64": "https://somewhere"},
    {"input": ""}, {"input": "this has too many words to replace"}, {"ref_text": ""},
    {"response_format": "mp3"},
    {"ref_audio_b64": base64.b64encode(wav_bytes(.5)).decode()},
    {"ref_audio_b64": base64.b64encode(wav_bytes(3)[:100]).decode()},
])
def test_companion_rejects_invalid_or_unsafe_requests(updates):
    assert fake_speech_app().test_client().post("/v1/audio/speech", json=speech_payload(**updates)).status_code == 400


def test_model_failure_is_an_explicit_service_error():
    def broken(*args):
        raise RuntimeError("model problem")
    client = create_app(loader=broken).test_client()
    response = client.post("/v1/audio/speech", json=speech_payload())
    assert response.status_code == 503
    assert "muted" in response.json["error"]


def test_render_flags_and_env_do_not_put_shared_token_in_command():
    from webapp import jobs

    job = {"id": 4, "video_path": "/movie.mp4", "edl_path": "/edl.json",
           "output_path": "/movie.clean.mp4", "options": json.dumps({
               "profanity_audio": "replace", "speech_host": "http://192.168.1.20:8765",
               "speech_model": DEFAULT_MODEL, "audio_track": 1, "subtitle_mode": "none",
           })}
    command = jobs.build_render_command(job)
    assert command[command.index("--profanity-audio") + 1] == "replace"
    assert command[command.index("--audio-track") + 1] == "1"
    assert "--speech-token" not in command
    with patch("webapp.jobs.settings_store.load", return_value={"speech_token": "secret"}):
        assert jobs._child_env()["CLEANCUT_SPEECH_TOKEN"] == "secret"


@pytest.fixture
def ffmpeg_available():
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("Real speech rendering requires FFmpeg and ffprobe")


def test_real_word_fit_rejects_excessive_stretch_and_keeps_exact_duration(tmp_path, ffmpeg_available):
    raw, fitted = tmp_path / "raw.wav", tmp_path / "fitted.wav"
    raw.write_bytes(wav_bytes(.6))
    _fit_clip(raw, fitted, .5)
    duration, rms = _wav_stats(fitted)
    assert abs(duration - .5) < .005
    assert rms > .002
    with pytest.raises(ValueError, match="stretching"):
        _fit_clip(raw, fitted, .1)


@pytest.mark.parametrize("explicit_action", [False, True])
def test_real_render_selects_english_track_fits_word_and_shifts_past_cut(tmp_path, ffmpeg_available, explicit_action):
    from cleancut.pipeline import PipelineOptions, render
    from cleancut.probe import audio_streams, probe_duration, probe_streams

    video = tmp_path / "movie.mkv"
    subprocess.run([
        "ffmpeg", "-nostdin", "-y", "-v", "error", "-f", "lavfi", "-i",
        "color=c=black:s=160x90:r=24:d=6", "-f", "lavfi", "-i",
        "sine=frequency=220:duration=6", "-f", "lavfi", "-i", "sine=frequency=440:duration=6",
        "-map", "0:v", "-map", "1:a", "-map", "2:a", "-metadata:s:a:0", "language=fra",
        "-metadata:s:a:1", "language=eng", "-c:v", "libx264", "-c:a", "aac", str(video),
    ], check=True)
    cfg = Config(profanity_audio="mute" if explicit_action else "replace", encoder="libx264", render_validation="full")
    cut = EditDecision(.5, 1.5, "cut", "nudity")
    edl = EditDecisionList(decisions=[cut, word_decision(action="replace" if explicit_action else "mute")])
    if explicit_action:
        edl.add(word_decision(4.2, 4.8))  # this ordinary Mute must never synthesize
    subs = [Subtitle(1, 0, 4, "That was a damn good idea.")]
    out = tmp_path / "out.mp4"
    received = []

    def request(_config, endpoint, payload=None, **kwargs):
        received.append(payload)
        assert endpoint == "/v1/audio/speech"
        assert "ref_audio" not in payload  # remote file paths would fail on a separate Mac.
        with wave.open(io.BytesIO(base64.b64decode(payload["ref_audio_b64"]))) as wav:
            assert wav.getframerate() == 24000
        return wav_bytes(.6, frequency=1000)

    with patch("cleancut.speech.check_service", return_value={}), \
         patch("cleancut.speech._request", side_effect=request):
        render(edl, subs, PipelineOptions(video=video, output=out, burn_subs=False), cfg)
    assert received[0]["input"] == "darn"
    assert len(received) == 1
    assert abs(probe_duration(out) - 5) < .15
    assert len(audio_streams(probe_streams(out))) == 1
    assert out.exists()
    assert not (tmp_path / ".cleancut_work").exists()
    # The generated 1000 Hz word lives at 1.2–1.8s after the one-second cut.
    # Uncensored audio outside it must be the selected 440 Hz English track.
    def dominant_at(start):
        import numpy as np

        data = subprocess.check_output(["ffmpeg", "-v", "error", "-ss", str(start),
            "-i", str(out), "-t", "0.15", "-ac", "1", "-ar", "24000", "-f", "s16le", "-"])
        samples = np.frombuffer(data, dtype="<i2")
        spectrum = abs(np.fft.rfft(samples))
        return np.argmax(spectrum) * 24000 / len(samples)
    assert abs(dominant_at(1.35) - 1000) < 20
    assert abs(dominant_at(3) - 440) < 20
    if explicit_action:
        import numpy as np

        muted = subprocess.check_output(["ffmpeg", "-v", "error", "-ss", "3.3", "-i", str(out),
            "-t", "0.15", "-ac", "1", "-ar", "24000", "-f", "s16le", "-"])
        assert np.sqrt(np.mean(np.frombuffer(muted, dtype="<i2").astype(float)**2)) < 20


def test_mix_command_keeps_video_copy_and_handles_soft_subtitle_input(tmp_path):
    from cleancut.editor import apply_mutes_and_subs

    subs = tmp_path / "subs.srt"
    subs.write_text("1\n00:00:00,000 --> 00:00:03,000\nHello\n")
    clip = SpeechClip(1.2, 1.8, tmp_path / "speech.wav")
    with patch("cleancut.editor._require_ffmpeg"), \
         patch("cleancut.editor._source_video_stream", return_value=None), \
         patch("cleancut.editor._source_video_format", return_value=("h264", "yuv420p")), \
         patch("cleancut.editor.subprocess.run") as run:
        apply_mutes_and_subs(tmp_path / "movie.mp4", [Range(1.2, 1.8)], subs,
                             tmp_path / "out.mp4", burn_subs=False, speech_clips=[clip], audio_index=2)
    command = run.call_args.args[0]
    graph = command[command.index("-filter_complex") + 1]
    assert "[0:2]volume" in graph and "[2:a]" in graph
    assert "normalize=0" in graph and "adelay=1200:all=1" in graph
    assert command[command.index("-c:v") + 1] == "copy"
    assert "0:a" not in command


def test_real_batching_bounds_inputs_and_renders_all_words(tmp_path, ffmpeg_available):
    from cleancut.editor import _batch_speech_clips
    from cleancut.probe import probe_duration

    word = tmp_path / "word.wav"
    word.write_bytes(wav_bytes(.1))
    clips = [SpeechClip(.5 + i * .11, .6 + i * .11, word) for i in range(30)]
    beds = _batch_speech_clips(clips, tmp_path)
    assert len(beds) == 2
    assert all(bed.start == 0 and bed.path.exists() for bed in beds)
    assert abs(probe_duration(beds[0].path) - clips[23].end) < .05
    assert abs(probe_duration(beds[1].path) - clips[-1].end) < .05


def test_mixing_failure_retries_with_word_mutes(tmp_path):
    from cleancut.editor import apply_mutes_and_subs

    with patch("cleancut.editor._require_ffmpeg"), \
         patch("cleancut.editor._audio_encoder_args", return_value=["-c:a", "aac"]), \
         patch("cleancut.editor._source_video_stream", return_value=None), \
         patch("cleancut.editor._source_video_format", return_value=("h264", "yuv420p")), \
         patch("cleancut.editor.subprocess.run", side_effect=[
             subprocess.CalledProcessError(1, "ffmpeg"), None,
         ]) as run:
        apply_mutes_and_subs(tmp_path / "in.mp4", [Range(1, 2)], None, tmp_path / "out.mp4",
                             burn_subs=False, speech_clips=[SpeechClip(1, 2, tmp_path / "word.wav")],
                             audio_index=1)
    assert run.call_count == 2
    assert "-filter_complex" in run.call_args_list[0].args[0]
    assert "-filter_complex" not in run.call_args_list[1].args[0]
    assert "between(t,1.000,2.000)" in run.call_args_list[1].args[0][
        run.call_args_list[1].args[0].index("-af") + 1]


def test_cached_preview_is_reused_at_render_and_shifted_without_regeneration(tmp_path):
    video = tmp_path / "movie"
    video.write_bytes(b"source-fingerprint")
    edl = EditDecisionList(decisions=[word_decision()])
    subs = [Subtitle(1, 0, 4, "That is a damn good idea.")]
    config = Config(profanity_audio="replace")

    def extract(*args):
        # The real FFmpeg interval-fit test above covers the remaining work.
        Path(args[-1]).write_bytes(wav_bytes(4))

    with patch("cleancut.speech.check_service", return_value={}), \
         patch("cleancut.speech._request", return_value=wav_bytes(.6)) as request, \
         patch("cleancut.speech._run_ffmpeg", side_effect=extract), \
         patch("cleancut.speech._fit_clip", side_effect=lambda raw, out, target, rms:
               out.write_bytes(wav_bytes(target))):
        preview = prepare_replacements(video, edl, subs, config, tmp_path / "preview",
                                       audio_index=1, cache_dir=tmp_path / "cache")
        edl.add(EditDecision(.5, 1.5, "cut", "nudity"))
        edl.add(EditDecision(.8, 1.2, "cut", "violence"))  # overlap is removed only once
        rendered = prepare_replacements(video, edl, subs, config, tmp_path / "render",
                                        audio_index=1, cache_dir=tmp_path / "cache")
    assert request.call_count == 1
    assert preview[0].path == rendered[0].path
    assert abs(rendered[0].start - 1.2) < .001


def test_word_at_exact_cut_end_survives_and_is_shifted_to_join(tmp_path):
    video = tmp_path / "movie"
    video.write_bytes(b"fingerprint")
    edl = EditDecisionList(decisions=[word_decision(), EditDecision(0, 2.2, "cut", "nudity")])

    def extract(*args):
        Path(args[-1]).write_bytes(wav_bytes(4))

    with patch("cleancut.speech.check_service", return_value={}), \
         patch("cleancut.speech._request", return_value=wav_bytes(.6)), \
         patch("cleancut.speech._run_ffmpeg", side_effect=extract), \
         patch("cleancut.speech._fit_clip", side_effect=lambda raw, out, target, rms:
               out.write_bytes(wav_bytes(target))):
        clips = prepare_replacements(video, edl, [Subtitle(1, 0, 4, "That was damn good.")],
                                     Config(profanity_audio="replace"), tmp_path / "speech")
    assert len(clips) == 1
    assert clips[0].start == 0
    assert abs(clips[0].end - .6) < .001


def test_web_connection_check_uses_unsaved_host_and_token_without_returning_secret():
    from webapp.app import app

    with patch("cleancut.speech.check_service", return_value={"model": DEFAULT_MODEL}) as check:
        response = app.test_client().post("/api/speech", json={
            "speech_host": "http://192.168.1.20:8765", "speech_model": DEFAULT_MODEL,
            "speech_token": "secret",
        })
    assert response.json["ok"]
    assert check.call_args.args[0].speech_token == "secret"
    assert "secret" not in response.get_data(as_text=True)


def test_companion_busy_response_is_bounded_not_parallel_inference():
    busy_results = []
    client = None

    def generate(*args):
        busy_results.append(client.post("/v1/audio/speech", json=speech_payload()).status_code)
        return wav_bytes(.6)

    client = create_app(loader=lambda name: object(), generator=generate).test_client()
    assert client.post("/v1/audio/speech", json=speech_payload()).status_code == 200
    assert busy_results == [503]
