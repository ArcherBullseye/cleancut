"""Fail-closed word separation and real soundtrack mixing regressions."""
import io
import json
import shutil
import subprocess
import sys
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from cleancut.background import (
    MAX_WINDOW,
    SAMPLE_RATE,
    check_runtime,
    eligible_mutes,
    prepare_background,
    valid_clip,
    window_groups,
)
from cleancut.config import Config
from cleancut.editor_ranges import Range
from cleancut.edl import EditDecision, EditDecisionList
from cleancut.separation_worker import read_window, residual, run, speech_detected, write_clip


def decision(start=2, end=2.4, **changes):
    values = {"start": start-.04, "end": end+.04, "action": "mute", "category": "profanity",
              "source": "whisper-word", "word_edits": [{"start": start, "end": end,
                    "text_before": "damn", "text_after": "darn", "category": "profanity"}]}
    values.update(changes)
    return EditDecision(**values)


def pcm(duration, channels=2, frequency=200):
    t = np.arange(round(duration * SAMPLE_RATE)) / SAMPLE_RATE
    samples = np.tile(.1 * np.sin(2 * np.pi * frequency * t)[:, None], (1, channels))
    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes((samples * 32767).astype("<i2").tobytes())
    return out.getvalue()


@pytest.mark.parametrize("changes", [
    {"accepted": False}, {"action": "cut"}, {"word_edits": []},
    {"source": "audio-event"}, {"start": -1}, {"end": 20},
    {"word_edits": [{"start": 0, "end": 1, "category": "profanity", "text_before": "damn"}]},
    {"word_edits": [{"start": float("nan"), "end": 2.4}]},
])
def test_background_only_accepts_precise_reviewed_word_mutes(changes):
    assert eligible_mutes(EditDecisionList(decisions=[decision(**changes)]), []) == []


def test_background_cannot_restore_audio_in_another_censor_or_cut():
    d = decision()
    assert eligible_mutes(EditDecisionList(decisions=[d]), []) == [Range(d.start, d.end)]
    assert eligible_mutes(EditDecisionList(decisions=[d]), [Range(0, 2.1)]) == []
    overlapping = EditDecision(2.1, 3, "mute", "sex", source="audio-event")
    assert eligible_mutes(EditDecisionList(decisions=[d, overlapping]), []) == []
    overlapping.accepted = False
    assert eligible_mutes(EditDecisionList(decisions=[d, overlapping]), [])


def test_windows_merge_nearby_words_but_remain_bounded():
    targets = [Range(i, i+.3) for i in range(2, 90)]
    groups = window_groups(targets, 90)
    assert len(groups) > 1
    assert all(w.duration <= MAX_WINDOW for w, _ in groups)
    assert [t for _, ts in groups for t in ts] == targets
    assert window_groups([Range(2, 3), Range(8, 20)], 3)[0][0].end == 3


@pytest.mark.parametrize("words, expected", [
    ([{"word": "damn", "start": 2, "end": 2.3, "probability": .8}], True),
    ([{"word": "hello", "start": 2, "end": 2.3, "probability": .8}], True),
    ([{"word": "hello", "start": 0, "end": 1, "probability": .8}], False),
    ([{"word": "hello", "start": 3, "end": 4, "probability": .8}], False),
    ([], False),
])
def test_speech_guard_checks_any_word_in_target_not_only_profanity(words, expected):
    result = {"segments": [{"words": words}]}
    assert speech_detected(result, 1.96, 2.44) is expected


def test_unverifiable_transcript_is_not_assumed_safe():
    assert speech_detected({"segments": [{"text": "heard speech"}]}, 1, 2)
    with pytest.raises(TypeError):
        speech_detected({}, 1, 2)
    with pytest.raises(ValueError):
        speech_detected({"segments": [{"words": [{"start": float("nan"), "end": 2}]}]}, 1, 2)


@pytest.mark.parametrize("channels", [1, 2, 6, 8])
def test_residual_keeps_channel_order_scale_and_exact_sample_count(channels):
    torch = pytest.importorskip("torch")
    t = np.arange(1000, dtype=np.float32) / SAMPLE_RATE
    mix = np.column_stack([.1 * np.sin(2*np.pi*(200+i*30)*t) for i in range(channels)])
    calls = []

    def fake(model, audio, **kwargs):
        calls.append(kwargs)
        outputs = torch.zeros((1, 4, 2, 1000))
        outputs[0, 3] = audio[0] * .5
        return outputs

    result = residual(SimpleNamespace(sources=["drums", "bass", "other", "vocals"]), mix, apply=fake)
    assert result.shape == mix.shape
    assert len(calls) == (1 if channels == 2 else channels)
    assert np.max(np.abs(result - (mix-mix.mean(0)) * .5)) < .003
    assert all(c["device"] == "cpu" and c["split"] for c in calls)


def test_clip_write_guards_and_wave_validation(tmp_path):
    source = tmp_path / "source.wav"
    source.write_bytes(pcm(4, 6))
    window = read_window(source)
    out = tmp_path / "clip.wav"
    write_clip(out, window, 1.96, 2.44)
    assert valid_clip(out, .48, 6)
    assert not valid_clip(out, .4, 6)
    assert not valid_clip(out, .48, 2)
    assert not valid_clip(tmp_path / "missing", .48, 6)
    with pytest.raises(ValueError, match="Invalid"):
        write_clip(out, window, -1, .4)
    with pytest.raises(ValueError, match="clip"):
        write_clip(out, window * 20, 1.96, 2.44)
    source.write_bytes(pcm(20))
    with pytest.raises(ValueError, match="bounded"):
        read_window(source)


def test_worker_checks_channels_independently_and_rejects_only_affected_target(tmp_path):
    pytest.importorskip("torch")
    source, rejected, accepted = tmp_path / "source.wav", tmp_path / "rejected.wav", tmp_path / "accepted.wav"
    source.write_bytes(pcm(4, 2))
    calls = []

    def transcribe(audio, **kwargs):
        calls.append(kwargs)
        # Speech leaked through only channel 2: a stereo downmix could miss it.
        return {"segments": []} if len(calls) == 1 else {
            "segments": [{"words": [{"word": "damn", "start": 2, "end": 2.3, "probability": .9}]}]}

    manifest = {"language": "eng", "tasks": [{"source": str(source), "targets": [
        {"start": 1.96, "end": 2.44, "output": str(rejected)},
        {"start": .2, "end": .6, "output": str(accepted)},
    ]}]}
    # Dependency API stubs only; window decoding, per-channel verification,
    # leakage decisions, sample slicing and atomic WAV writing are real.
    modules = {"torchaudio.functional": SimpleNamespace(resample=lambda t, old, new: t),
               "whisper.tokenizer": SimpleNamespace(LANGUAGES={"en": "english"})}
    with patch.dict(sys.modules, modules), \
         patch("cleancut.separation_worker.residual", side_effect=lambda model, mix: mix):
        run(manifest, object(), SimpleNamespace(transcribe=transcribe))
    assert len(calls) == 2 and all(c["word_timestamps"] for c in calls)
    assert not rejected.exists()
    assert valid_clip(accepted, .4, 2)


def test_missing_installation_is_actionable_and_no_inference_is_attempted(tmp_path):
    with patch("cleancut.background.runtime_python", return_value=tmp_path / "missing"), \
         patch("cleancut.background.model_directory", return_value=tmp_path), \
         pytest.raises(ValueError, match="install-separation.sh"):
        check_runtime()
    video = tmp_path / "video"
    video.write_bytes(b"fingerprint")
    with patch("cleancut.background.check_runtime", side_effect=ValueError("not installed")), \
         patch("cleancut.background.subprocess.run") as worker:
        assert not prepare_background(video, EditDecisionList(decisions=[decision()]),
            Config(preserve_background=True), tmp_path / "work", audio_index=1, channels=2, cuts=[])
    worker.assert_not_called()


def test_verified_background_cache_reuses_source_coordinates_and_shifts_once(tmp_path):
    video = tmp_path / "video"
    video.write_bytes(b"fingerprint")
    config, edl = Config(preserve_background=True), EditDecisionList(decisions=[decision()])

    def worker(command, **kwargs):
        manifest = json.loads(Path(command[-1]).read_text())
        for task in manifest["tasks"]:
            for target in task["targets"]:
                Path(target["output"]).write_bytes(pcm(target["end"]-target["start"]))
    with patch("cleancut.background.check_runtime"), \
         patch("cleancut.background.probe_duration", return_value=5), \
         patch("cleancut.background._run_ffmpeg"), \
         patch("cleancut.background.subprocess.run", side_effect=worker) as run:
        clips = prepare_background(video, edl, config, tmp_path / "work", audio_index=1,
                                   channels=2, cuts=[], cache_dir=tmp_path / "cache")
        shifted = prepare_background(video, edl, config, tmp_path / "work2", audio_index=1,
            channels=2, cuts=[Range(.5, 1.5)], cache_dir=tmp_path / "cache")
    assert run.call_count == 1
    assert clips[0].path == shifted[0].path
    assert abs(shifted[0].start-.96) < .0001
    assert abs(shifted[0].end-1.44) < .0001


def test_runtime_error_or_no_verified_output_cannot_restore_original_audio(tmp_path):
    video = tmp_path / "video"
    video.write_bytes(b"fingerprint")
    with patch("cleancut.background.check_runtime"), \
         patch("cleancut.background.probe_duration", return_value=5), \
         patch("cleancut.background._run_ffmpeg"), \
         patch("cleancut.background.subprocess.run", side_effect=RuntimeError("worker failed")):
        assert not prepare_background(video, EditDecisionList(decisions=[decision()]),
            Config(preserve_background=True), tmp_path / "work", audio_index=1, channels=2, cuts=[])


def test_setting_is_boolean_and_reaches_render_command(cli_args, tmp_path):
    from cleancut.cli import _apply_common
    from webapp import jobs, settings

    with patch("webapp.settings._SETTINGS_PATH", tmp_path / "settings.json"):
        assert settings.save({"preserve_background": True})["preserve_background"] is True
        assert settings.save({"preserve_background": "false"})["preserve_background"] is True
    config = Config()
    _apply_common(cli_args("clean", "movie", "--preserve-background"), config)
    assert config.preserve_background
    job = {"video_path": "movie", "edl_path": "edl", "id": 12, "output_path": str(tmp_path / "out"),
           "options": json.dumps({"preserve_background": True})}
    assert "--preserve-background" in jobs.build_render_command(job)


def test_separation_health_endpoint_reports_errors_without_exposing_tracebacks():
    from webapp.app import app

    with patch("cleancut.background.check_runtime", side_effect=ValueError("install-separation.sh")):
        response = app.test_client().get("/api/separation")
    assert not response.json["ok"] and "install-separation.sh" in response.json["reason"]


def test_background_preview_uses_original_timeline_after_cuts(tmp_path):
    from cleancut.probe import Stream
    from cleancut.speech import SpeechClip
    from webapp.app import app

    source = tmp_path / "source"
    source.write_bytes(b"movie")
    path = tmp_path / "edl.json"
    edl = EditDecisionList(decisions=[decision(), EditDecision(.5, 1.5, "cut", "nudity")])
    edl.to_json(path)
    cached = tmp_path / "background.wav"
    cached.write_bytes(pcm(.48))
    (tmp_path / "background").mkdir()  # the stubbed preparer normally creates this cache
    job = {"video_path": str(source), "options": "{}"}

    def mix(video, target, output, **kwargs):
        assert target == Range(1.96, 2.44)
        assert kwargs["overlays"][0].start == 1.96  # not the edited .96s timestamp
        output.write_bytes(pcm(4.44))

    with patch("webapp.app._edl_job", return_value=(job, path, None)), \
         patch("webapp.app._video_for", return_value=source), \
         patch("cleancut.probe.probe_duration", return_value=5), \
         patch("cleancut.probe.probe_streams", return_value=[Stream(1, "aac", "audio", channels=2)]), \
         patch("webapp.app.settings_store.load", return_value={"profanity_audio": "mute"}), \
         patch("cleancut.background.prepare_background", return_value=[SpeechClip(.96, 1.44, cached)]) as prepare, \
         patch("cleancut.background.preview_mix", side_effect=mix):
        response = app.test_client().post("/api/job/1/background/0")
        assert response.status_code == 200 and response.mimetype == "audio/wav", response.json
        assert prepare.call_args.kwargs["only"] == Range(1.96, 2.44)
        missing = app.test_client().post("/api/job/1/background/99")
        assert missing.status_code == 404
        response.close()


@pytest.fixture
def real_ffmpeg():
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("Requires real FFmpeg")


@pytest.mark.parametrize("channels", [1, 2, 6, 8])
@pytest.mark.parametrize("replace", [False, True])
def test_real_mix_preserves_background_and_channel_count_after_cut(tmp_path, real_ffmpeg, channels, replace):
    from cleancut.pipeline import PipelineOptions, render
    from cleancut.probe import audio_streams, probe_duration, probe_streams
    from cleancut.speech import SpeechClip

    source = tmp_path / "source.mkv"
    soundtrack = tmp_path / "audio.wav"
    music = read_window_bytes(pcm(5, channels, 200))
    voice = read_window_bytes(pcm(5, channels, 440))
    write_pcm(soundtrack, music + voice)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
        "color=c=black:s=64x64:r=24:d=5", "-i", str(soundtrack),
        "-c:v", "libx264", "-c:a", "pcm_s16le", str(source)], check=True)
    word = tmp_path / "speech.wav"
    word.write_bytes(pcm(.4, 1, 1000))
    edl = EditDecisionList(decisions=[EditDecision(.5, 1.5, "cut", "nudity"), decision()])
    output = tmp_path / "out.mp4"
    actual_run = subprocess.run

    def worker(command, **kwargs):
        if "cleancut.separation_worker" not in command:
            return actual_run(command, **kwargs)
        manifest = json.loads(Path(command[-1]).read_text())
        for task in manifest["tasks"]:
            for target in task["targets"]:
                Path(target["output"]).write_bytes(pcm(target["end"]-target["start"], channels, 200))
        return SimpleNamespace(returncode=0)

    cfg = Config(preserve_background=True, profanity_audio="replace" if replace else "mute",
                 encoder="libx264", render_validation="full")
    with patch("cleancut.background.check_runtime"), \
         patch("cleancut.background.subprocess.run", side_effect=worker), \
         patch("cleancut.speech.prepare_replacements", return_value=[SpeechClip(1, 1.4, word)]):
        render(edl, [], PipelineOptions(video=source, output=output, burn_subs=False), cfg)
    tracks = audio_streams(probe_streams(output))
    assert len(tracks) == 1 and tracks[0].channels == channels
    assert abs(probe_duration(output)-4) < .08
    samples = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(output),
        "-map", "0:a:0", "-ar", str(SAMPLE_RATE), "-f", "s16le", "-"])
    rendered = np.frombuffer(samples, dtype="<i2").reshape(-1, channels).astype(float) / 32768
    part = rendered[round(1.1*SAMPLE_RATE):round(1.3*SAMPLE_RATE)]
    frequencies = np.fft.rfftfreq(len(part), 1/SAMPLE_RATE)
    spectrum = abs(np.fft.rfft(part, axis=0)) * 2 / len(part)
    for ch in range(channels):
        def amplitude(freq, ch=ch):
            return spectrum[np.argmin(abs(frequencies-freq)), ch]
        assert amplitude(200) > .07  # music survived the muted interval
        assert amplitude(440) < .003  # original spoken word did not
        if replace:
            assert spectrum[(frequencies > 980) & (frequencies < 1020), :].max() > .02


def read_window_bytes(data):
    with wave.open(io.BytesIO(data), "rb") as wav:
        return np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").reshape(-1, wav.getnchannels()).astype(np.float32)/32768


def write_pcm(path, samples):
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(samples.shape[1])
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes((samples*32767).astype("<i2").tobytes())


def test_real_overlay_batching_keeps_surround_losslessly(tmp_path, real_ffmpeg):
    from cleancut.editor import _batch_speech_clips
    from cleancut.probe import audio_streams, probe_streams
    from cleancut.speech import SpeechClip

    word, background = tmp_path / "word.wav", tmp_path / "background.wav"
    word.write_bytes(pcm(.1, 1))
    background.write_bytes(pcm(.1, 6))
    clips = [SpeechClip(.1, .2, word)] + [SpeechClip(.5+i*.11, .6+i*.11, background) for i in range(27)]
    beds = _batch_speech_clips(clips, tmp_path)
    assert len(beds) == 2
    assert all(b.path.suffix == ".flac" for b in beds)
    assert all(audio_streams(probe_streams(b.path))[0].channels == 6 for b in beds)


def test_real_mixed_preview_is_bounded_not_the_remaining_movie(tmp_path, real_ffmpeg):
    from cleancut.background import preview_mix
    from cleancut.probe import probe_duration
    from cleancut.speech import SpeechClip

    source, background, output = tmp_path / "movie.wav", tmp_path / "background.wav", tmp_path / "preview.wav"
    source.write_bytes(pcm(20))
    background.write_bytes(pcm(.48))
    preview_mix(source, Range(9.96, 10.44), output, audio_index=0,
                overlays=[SpeechClip(9.96, 10.44, background)])
    assert abs(probe_duration(output) - 4.48) < .002
