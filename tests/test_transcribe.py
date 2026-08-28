"""Tests for cleancut/transcribe.py.

All Whisper and torch calls are mocked — no GPU or model download required.
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from cleancut.transcribe import _autodetect_device, _resolve_device


# ---------------------------------------------------------------------------
# _resolve_device
# ---------------------------------------------------------------------------

class TestResolveDevice:
    def test_cpu_passthrough(self):
        result = _resolve_device("cpu", word_timestamps=False)
        assert result == "cpu"

    def test_cuda_passthrough(self):
        result = _resolve_device("cuda", word_timestamps=False)
        assert result == "cuda"

    def test_mps_without_word_timestamps_passthrough(self):
        result = _resolve_device("mps", word_timestamps=False)
        assert result == "mps"

    def test_mps_with_word_timestamps_falls_back_to_cpu(self, capsys):
        result = _resolve_device("mps", word_timestamps=True)
        assert result == "cpu"
        # Should print a warning
        captured = capsys.readouterr()
        assert "WARNING" in captured.out or "MPS" in captured.out

    def test_none_calls_autodetect(self):
        with patch("cleancut.transcribe._autodetect_device", return_value="cpu") as mock_auto:
            result = _resolve_device(None, word_timestamps=False)
        mock_auto.assert_called_once_with(word_timestamps=False)
        assert result == "cpu"

    def test_none_with_word_timestamps_calls_autodetect_with_flag(self):
        with patch("cleancut.transcribe._autodetect_device", return_value="cpu") as mock_auto:
            _resolve_device(None, word_timestamps=True)
        mock_auto.assert_called_once_with(word_timestamps=True)


# ---------------------------------------------------------------------------
# _autodetect_device
# ---------------------------------------------------------------------------

class TestAutodetectDevice:
    def test_returns_valid_device_string(self):
        result = _autodetect_device()
        assert result in ("cpu", "cuda", "mps")

    def test_falls_back_to_cpu_if_torch_missing(self):
        with patch.dict(sys.modules, {"torch": None}):
            result = _autodetect_device()
        assert result == "cpu"

    def test_prefers_mps_when_available_and_no_word_timestamps(self):
        mock_torch = MagicMock()
        mock_torch.backends.mps.is_available.return_value = True
        mock_torch.cuda.is_available.return_value = False
        with patch.dict(sys.modules, {"torch": mock_torch}):
            result = _autodetect_device(word_timestamps=False)
        assert result == "mps"

    def test_mps_with_word_timestamps_returns_cpu(self):
        mock_torch = MagicMock()
        mock_torch.backends.mps.is_available.return_value = True
        with patch.dict(sys.modules, {"torch": mock_torch}):
            result = _autodetect_device(word_timestamps=True)
        assert result == "cpu"

    def test_prefers_cuda_when_mps_unavailable(self):
        mock_torch = MagicMock()
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.cuda.is_available.return_value = True
        with patch.dict(sys.modules, {"torch": mock_torch}):
            result = _autodetect_device()
        assert result == "cuda"

    def test_cpu_when_neither_mps_nor_cuda(self):
        mock_torch = MagicMock()
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.cuda.is_available.return_value = False
        with patch.dict(sys.modules, {"torch": mock_torch}):
            result = _autodetect_device()
        assert result == "cpu"


# ---------------------------------------------------------------------------
# transcribe()
# ---------------------------------------------------------------------------

def _make_whisper_result(segments=None):
    """Build a fake Whisper transcribe() result dict."""
    if segments is None:
        segments = [
            {
                "start": 0.5,
                "end": 3.2,
                "text": "Hello world",
                "words": [
                    {"word": "Hello", "start": 0.5, "end": 1.0, "probability": 0.99},
                    {"word": "world", "start": 1.1, "end": 1.8, "probability": 0.97},
                ],
            },
        ]
    return {"segments": segments}


class TestTranscribe:
    def _mock_whisper(self, result=None):
        """Return a mock whisper module with load_model returning a mock model."""
        if result is None:
            result = _make_whisper_result()
        mock_model = MagicMock()
        mock_model.transcribe.return_value = result
        mock_whisper = MagicMock()
        mock_whisper.load_model.return_value = mock_model
        return mock_whisper, mock_model

    def test_returns_subtitles_and_words(self, tmp_path):
        from cleancut.transcribe import transcribe
        fake_video = tmp_path / "v.mp4"
        fake_video.write_bytes(b"\x00")
        mock_whisper, mock_model = self._mock_whisper()

        with patch.dict(sys.modules, {"whisper": mock_whisper}):
            subs, words = transcribe(fake_video, model_name="base", device="cpu")

        assert len(subs) == 1
        assert subs[0].text == "Hello world"
        assert abs(subs[0].start - 0.5) < 0.01
        assert abs(subs[0].end - 3.2) < 0.01
        assert len(words) == 2
        assert words[0].text == "Hello"
        assert words[1].text == "world"

    def test_word_timestamps_false_returns_empty_words(self, tmp_path):
        from cleancut.transcribe import transcribe
        fake_video = tmp_path / "v.mp4"
        fake_video.write_bytes(b"\x00")
        result = _make_whisper_result(segments=[
            {"start": 0.0, "end": 2.0, "text": "Test", "words": [
                {"word": "Test", "start": 0.0, "end": 1.0, "probability": 0.9},
            ]},
        ])
        mock_whisper, mock_model = self._mock_whisper(result)

        with patch.dict(sys.modules, {"whisper": mock_whisper}):
            subs, words = transcribe(
                fake_video, model_name="base", device="cpu", word_timestamps=False
            )

        # word_timestamps=False is passed to model.transcribe; our code
        # only populates the words list when word_timestamps=True.
        assert len(subs) == 1
        assert len(words) == 0

    def test_empty_segments_returns_empty_lists(self, tmp_path):
        from cleancut.transcribe import transcribe
        fake_video = tmp_path / "v.mp4"
        fake_video.write_bytes(b"\x00")
        mock_whisper, _ = self._mock_whisper(_make_whisper_result(segments=[]))

        with patch.dict(sys.modules, {"whisper": mock_whisper}):
            subs, words = transcribe(fake_video, model_name="base", device="cpu")

        assert subs == []
        assert words == []

    def test_load_model_called_with_correct_args(self, tmp_path):
        from cleancut.transcribe import transcribe
        fake_video = tmp_path / "v.mp4"
        fake_video.write_bytes(b"\x00")
        mock_whisper, _ = self._mock_whisper()

        with patch.dict(sys.modules, {"whisper": mock_whisper}):
            transcribe(fake_video, model_name="small", device="cpu")

        mock_whisper.load_model.assert_called_once_with("small", device="cpu")

    def test_audio_path_is_used_when_provided(self, tmp_path):
        from cleancut.transcribe import transcribe
        fake_video = tmp_path / "v.mp4"
        fake_video.write_bytes(b"\x00")
        fake_audio = tmp_path / "a.wav"
        fake_audio.write_bytes(b"\x00")
        mock_whisper, mock_model = self._mock_whisper()

        with patch.dict(sys.modules, {"whisper": mock_whisper}):
            transcribe(fake_video, model_name="base", device="cpu",
                       audio_path=fake_audio)

        # The model.transcribe call should use the audio path string, not the video.
        call_args = mock_model.transcribe.call_args
        assert call_args[0][0] == str(fake_audio)

    def test_raises_runtime_error_when_whisper_not_installed(self, tmp_path):
        from cleancut.transcribe import transcribe
        fake_video = tmp_path / "v.mp4"
        fake_video.write_bytes(b"\x00")

        # Simulate whisper not being importable.
        with patch.dict(sys.modules, {"whisper": None}):
            with pytest.raises(RuntimeError, match="Whisper not installed"):
                transcribe(fake_video, model_name="base", device="cpu")

    def test_skips_empty_word_tokens(self, tmp_path):
        from cleancut.transcribe import transcribe
        fake_video = tmp_path / "v.mp4"
        fake_video.write_bytes(b"\x00")
        result = _make_whisper_result(segments=[
            {
                "start": 0.0,
                "end": 2.0,
                "text": "Hi",
                "words": [
                    {"word": "", "start": 0.0, "end": 0.1, "probability": 0.5},  # blank, skip
                    {"word": "  ", "start": 0.1, "end": 0.2, "probability": 0.5},  # spaces, skip
                    {"word": "Hi", "start": 0.2, "end": 0.8, "probability": 0.9},
                ],
            },
        ])
        mock_whisper, _ = self._mock_whisper(result)

        with patch.dict(sys.modules, {"whisper": mock_whisper}):
            _, words = transcribe(fake_video, model_name="base", device="cpu")

        assert len(words) == 1
        assert words[0].text == "Hi"

    def test_language_param_forwarded_to_transcribe(self, tmp_path):
        from cleancut.transcribe import transcribe
        fake_video = tmp_path / "v.mp4"
        fake_video.write_bytes(b"\x00")
        mock_whisper, mock_model = self._mock_whisper()

        with patch.dict(sys.modules, {"whisper": mock_whisper}):
            transcribe(fake_video, model_name="base", device="cpu", language="fr")

        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs.get("language") == "fr"
