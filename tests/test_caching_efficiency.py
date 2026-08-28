"""Regression tests for caching and efficiency fixes.

The expensive stages (Whisper, LLM, VLM) must cache like the cheap ones
already do, and different configs must not evict each other's entries.
"""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from cleancut import cache
from cleancut.config import Config
from cleancut.edl import EditDecisionList
from cleancut.scenes import Shot
from cleancut.subtitles import Subtitle
from cleancut.transcribe import Word


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "cache")
    yield


VALID_SRT = "1\n00:00:00,000 --> 00:00:02,000\nhello world\n"


class TestCacheKeyIsolation:
    def test_different_config_hashes_coexist(self, tmp_path):
        """Toggling between two configs must not evict each other's entries."""
        v = tmp_path / "movie.mp4"
        v.write_bytes(b"abc")
        cache.save(v, "feat", "aaa", {"v": 1})
        cache.save(v, "feat", "bbb", {"v": 2})
        assert cache.load(v, "feat", "aaa") == {"v": 1}
        assert cache.load(v, "feat", "bbb") == {"v": 2}

    def test_clear_removes_all_hashes_for_feature(self, tmp_path):
        v = tmp_path / "movie.mp4"
        v.write_bytes(b"abc")
        cache.save(v, "feat", "aaa", {"v": 1})
        cache.save(v, "feat", "bbb", {"v": 2})
        assert cache.clear(v, "feat") == 2


class TestWhisperCache:
    def _run(self, tmp_path, mock_transcribe):
        from cleancut.pipeline import PipelineOptions, _get_subtitles_and_words
        from cleancut.probe import Stream

        video = tmp_path / "movie.mp4"
        video.write_bytes(b"\x00" * 64)
        streams = [Stream(index=1, codec_name="aac", codec_type="audio", language="eng")]

        def fake_extract(v, idx):
            wav = tmp_path / "extracted.wav"
            wav.write_bytes(b"")
            return wav

        opts = PipelineOptions(video=video)
        config = Config.load_defaults()
        with patch("cleancut.probe.probe_streams", return_value=streams), \
             patch("cleancut.probe.find_sidecar_subtitle", return_value=None), \
             patch("cleancut.probe.pick_embedded_subtitle", return_value=None), \
             patch("cleancut.probe.extract_audio_to_wav", side_effect=fake_extract) as mock_ext, \
             patch("cleancut.transcribe.transcribe", mock_transcribe):
            first = _get_subtitles_and_words(opts, config)
            second = _get_subtitles_and_words(opts, config)
        return first, second, mock_ext

    def test_second_run_uses_cache(self, tmp_path):
        fake_subs = [Subtitle(index=1, start=0.0, end=2.0, text="hello")]
        fake_words = [Word(start=0.0, end=0.5, text="hello", probability=0.9)]
        mock_transcribe = MagicMock(return_value=(fake_subs, fake_words))

        first, second, mock_extract = self._run(tmp_path, mock_transcribe)

        assert mock_transcribe.call_count == 1
        assert mock_extract.call_count == 1  # no audio decode on cache hit either
        assert second == first
        assert second[1][0].text == "hello"


class TestWordsSidecarReadBack:
    def test_words_json_next_to_explicit_subs_is_loaded(self, tmp_path):
        """--save-transcript writes a .words.json; a rerun with --subs on that
        transcript must get its word-level precision back."""
        from cleancut.pipeline import PipelineOptions, _get_subtitles_and_words

        srt = tmp_path / "movie.whisper.srt"
        srt.write_text(VALID_SRT)
        (tmp_path / "movie.whisper.words.json").write_text(
            '[{"start": 0.0, "end": 0.5, "text": "hello", "probability": 0.9}]'
        )
        opts = PipelineOptions(video=tmp_path / "movie.mp4", subs=srt)
        subs, words = _get_subtitles_and_words(opts, Config.load_defaults())
        assert len(subs) == 1
        assert len(words) == 1
        assert words[0].text == "hello"


class _CuttingClient:
    """Ollama client that flags everything as a confident sex scene."""

    def generate(self, **kwargs):
        return {"response": "ok"}

    def chat(self, **kwargs):
        return {"message": {"content":
            '{"category": "sex", "should_cut": true, "confidence": 0.9, "reasoning": "x"}'}}


class _ExplodingClientFactory:
    """make_ollama_client replacement that fails the test if ever called."""

    def __call__(self, host):
        raise AssertionError("Ollama client created despite cache hit")


class TestLLMCache:
    def test_second_run_skips_ollama_entirely(self, tmp_path):
        from cleancut.classify_dialogue import LLMParams, classify_dialogue

        video = tmp_path / "movie.mp4"
        video.write_bytes(b"\x00")
        subs = [Subtitle(index=1, start=0.0, end=2.0, text="line one"),
                Subtitle(index=2, start=3.0, end=5.0, text="line two")]

        with patch("cleancut.classify_dialogue.make_ollama_client",
                   return_value=_CuttingClient()):
            first = classify_dialogue(subs, LLMParams(), video=video)
        assert len(first) == 1

        with patch("cleancut.classify_dialogue.make_ollama_client",
                   new=_ExplodingClientFactory()):
            second = classify_dialogue(subs, LLMParams(), video=video)
        assert [(d.start, d.end, d.category) for d in second] == \
            [(d.start, d.end, d.category) for d in first]


class TestVLMCache:
    def test_second_run_skips_ollama_entirely(self, tmp_path):
        from cleancut.classify_visual import VLMParams, scan_with_vlm

        video = tmp_path / "movie.mp4"
        video.write_bytes(b"\x00")
        shots = [Shot(0.0, 10.0)]
        frame = tmp_path / "frame.jpg"
        frame.write_bytes(b"\xff")

        vlm_result = {"explicit": True, "confidence": 0.9, "description": "d"}
        with patch("cleancut.classify_visual.make_ollama_client",
                   return_value=_CuttingClient()), \
             patch("cleancut.classify_visual._extract_frame", return_value=frame), \
             patch("cleancut.classify_visual._classify_frame", return_value=vlm_result):
            first = scan_with_vlm(video, shots, [], EditDecisionList(), VLMParams(mode="all"))
        assert len(first) == 1

        with patch("cleancut.classify_visual.make_ollama_client",
                   new=_ExplodingClientFactory()):
            second = scan_with_vlm(video, shots, [], EditDecisionList(), VLMParams(mode="all"))
        assert [(d.start, d.end) for d in second] == [(d.start, d.end) for d in first]


class TestRunFullEdlFastPath:
    def test_no_subtitle_resolution_when_not_burning(self, tmp_path):
        """cleancut clean --edl X --no-burn-subs must never trigger Whisper
        just to compute subtitles it will not use."""
        from cleancut.pipeline import PipelineOptions, run_full

        video = tmp_path / "movie.mp4"
        video.write_bytes(b"\x00")
        edl_path = tmp_path / "movie.edl.json"
        EditDecisionList(video_path=str(video)).to_json(edl_path)

        opts = PipelineOptions(video=video, edl_in=edl_path, burn_subs=False)
        with patch("cleancut.pipeline._get_subtitles_and_words") as mock_get:
            run_full(opts, Config.load_defaults())
        mock_get.assert_not_called()


class TestAudioEventsBatching:
    def _params(self):
        from cleancut.audio_events import AudioEventParams
        return AudioEventParams(use_cache=False)

    def test_clips_are_classified_in_batches(self, tmp_path):
        from cleancut import audio_events

        video = tmp_path / "movie.mp4"
        video.write_bytes(b"\x00")
        shots = [Shot(float(i * 10), float(i * 10 + 8)) for i in range(20)]
        wav = tmp_path / "a.wav"
        wav.write_bytes(b"")

        batch_sizes: list[int] = []

        def fake_classify(extractor, model, device, clips):
            batch_sizes.append(len(clips))
            return [[("Moaning", 0.9)] for _ in clips]

        with patch.object(audio_events, "_extract_full_audio_to_wav", return_value=wav), \
             patch.object(audio_events, "_load_audio",
                          return_value=(np.zeros(16000 * 250, dtype=np.float32), 16000)), \
             patch.object(audio_events, "_load_model",
                          return_value=(MagicMock(), MagicMock(), "cpu")), \
             patch.object(audio_events, "_classify_clips", side_effect=fake_classify):
            edl = audio_events.scan_audio_events(video, shots, self._params())

        assert sum(batch_sizes) == 20
        assert all(s <= audio_events.AST_BATCH_SIZE for s in batch_sizes)
        assert len(batch_sizes) < 20  # actually batched, not one-by-one
        assert len(edl) == 20
        assert all(d.category == "sex" for d in edl.decisions)

    def test_skip_violence_drops_violence_only_hits(self, tmp_path):
        from cleancut import audio_events

        video = tmp_path / "movie.mp4"
        video.write_bytes(b"\x00")
        shots = [Shot(0.0, 8.0)]
        wav = tmp_path / "a.wav"
        wav.write_bytes(b"")

        with patch.object(audio_events, "_extract_full_audio_to_wav", return_value=wav), \
             patch.object(audio_events, "_load_audio",
                          return_value=(np.zeros(16000 * 20, dtype=np.float32), 16000)), \
             patch.object(audio_events, "_load_model",
                          return_value=(MagicMock(), MagicMock(), "cpu")), \
             patch.object(audio_events, "_classify_clips",
                          return_value=[[("Screaming", 0.95)]]):
            edl = audio_events.scan_audio_events(video, shots, self._params())

        assert len(edl) == 0


class TestPipelinePassesAudioTrack:
    def test_audio_events_get_selected_track_index(self, tmp_path):
        """AST must analyze the same track Whisper transcribes, not ffmpeg's
        default stream (which can be a commentary or foreign dub)."""
        from cleancut.pipeline import PipelineOptions, build_edl
        from cleancut.probe import Stream

        video = tmp_path / "movie.mp4"
        video.write_bytes(b"\x00")
        config = Config.load_defaults()
        config.audio_events_enabled = True
        config.llm_enabled = False
        config.vlm_enabled = False
        config.density_enabled = False

        streams = [
            Stream(index=1, codec_name="aac", codec_type="audio", language="spa"),
            Stream(index=2, codec_name="aac", codec_type="audio", language="eng"),
        ]
        captured = {}

        def fake_scan(video, shots, params, audio_track_index=None):
            captured["idx"] = audio_track_index
            return EditDecisionList()

        opts = PipelineOptions(video=video, use_visual=False, use_whisper=False)
        with patch("cleancut.pipeline._get_subtitles_and_words", return_value=([], [])), \
             patch("cleancut.pipeline._detect_scenes_if_enabled",
                   return_value=[Shot(0.0, 10.0)]), \
             patch("cleancut.probe.probe_streams", return_value=streams), \
             patch("cleancut.audio_events.scan_audio_events", side_effect=fake_scan):
            build_edl(opts, config)

        assert captured["idx"] == 2  # the English track, matching Whisper's pick


class TestScenesEmptyCache:
    def test_empty_shot_list_is_cached(self, tmp_path):
        import sys
        from cleancut.scenes import detect_shots

        video = tmp_path / "movie.mp4"
        video.write_bytes(b"\x00")

        mock_scenedetect = MagicMock()
        mock_scenedetect.detect.return_value = []
        with patch.dict(sys.modules, {"scenedetect": mock_scenedetect}), \
             patch("cleancut.cache.load", return_value=None), \
             patch("cleancut.cache.save") as mock_save:
            shots = detect_shots(video, use_cache=True)

        assert shots == []
        mock_save.assert_called_once()


class TestTempFileHygiene:
    def test_detect_on_frame_fallback_unlinks_temp_jpg(self):
        from cleancut.visual import _detect_on_frame

        captured = {}

        class FakeDetector:
            def detect(self, arg):
                if isinstance(arg, str):
                    captured["path"] = arg
                    return []
                raise TypeError("this build wants a path")

        class FakeCv2:
            @staticmethod
            def imwrite(name, frame):
                Path(name).write_bytes(b"jpg")

        result = _detect_on_frame(FakeDetector(), FakeCv2, frame=object())
        assert result == []
        assert not Path(captured["path"]).exists()

    def test_extract_audio_to_wav_does_not_leak_fd(self, tmp_path):
        from cleancut.probe import extract_audio_to_wav

        with patch("shutil.which", return_value="/usr/bin/ffmpeg"), \
             patch("subprocess.run"):
            before = len(os.listdir("/dev/fd"))
            out = extract_audio_to_wav(tmp_path / "v.mp4", 1)
            after = len(os.listdir("/dev/fd"))
        out.unlink(missing_ok=True)
        assert after == before
