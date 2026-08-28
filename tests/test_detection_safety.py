"""Regression tests: detection stages must fail loudly, not silently report nothing.

A content-filtering tool that completes a run with zero detections because
Ollama was down is indistinguishable from a clean movie — these tests pin the
fail-loud behavior.
"""
from __future__ import annotations

import json
import subprocess
import sys
from unittest.mock import patch

import pytest

from cleancut.classify_dialogue import LLMParams, classify_dialogue
from cleancut.classify_visual import VLMParams, _flagged_categories, scan_with_vlm
from cleancut.scenes import Shot
from cleancut.subtitles import Subtitle


def _subs_in_chunks(n_chunks: int) -> list[Subtitle]:
    """n_chunks chunks of two lines each, separated by long silence gaps."""
    subs = []
    for i in range(n_chunks):
        base = i * 200.0
        subs.append(Subtitle(index=2 * i + 1, start=base, end=base + 2, text="line one"))
        subs.append(Subtitle(index=2 * i + 2, start=base + 3, end=base + 5, text="line two"))
    return subs


class _DeadClient:
    """Ollama server unreachable: every call errors."""

    def generate(self, **kwargs):
        raise ConnectionError("connection refused")

    def chat(self, **kwargs):
        raise ConnectionError("connection refused")


class _WarmButFailingClient:
    """Model loads, but every chat call errors (e.g. model OOMs / server dies)."""

    def __init__(self):
        self.chat_calls = 0

    def generate(self, **kwargs):
        return {"response": "ok"}

    def chat(self, **kwargs):
        self.chat_calls += 1
        raise ConnectionError("boom")


class _FixedResponseClient:
    def __init__(self, payload: dict):
        self.payload = payload

    def generate(self, **kwargs):
        return {"response": "ok"}

    def chat(self, **kwargs):
        return {"message": {"content": json.dumps(self.payload)}}


class TestClassifyDialogueFailLoud:
    def test_raises_when_ollama_unreachable(self):
        with patch("cleancut.classify_dialogue.make_ollama_client", return_value=_DeadClient()):
            with pytest.raises(RuntimeError, match="[Oo]llama"):
                classify_dialogue(_subs_in_chunks(2), LLMParams())

    def test_aborts_after_consecutive_chat_failures(self):
        client = _WarmButFailingClient()
        with patch("cleancut.classify_dialogue.make_ollama_client", return_value=client):
            with pytest.raises(RuntimeError):
                classify_dialogue(_subs_in_chunks(20), LLMParams())
        # Must abort early, not grind through every chunk eating timeouts.
        assert client.chat_calls < 20

    def test_malformed_confidence_skips_chunk_instead_of_crashing(self):
        payload = {"category": "sex", "should_cut": True,
                   "confidence": "high", "reasoning": "x"}
        client = _FixedResponseClient(payload)
        with patch("cleancut.classify_dialogue.make_ollama_client", return_value=client):
            edl = classify_dialogue(_subs_in_chunks(2), LLMParams())
        assert len(edl) == 0


class TestScanWithVlmFailLoud:
    def test_raises_when_ollama_unreachable(self, tmp_path):
        video = tmp_path / "movie.mp4"
        video.write_bytes(b"\x00")
        shots = [Shot(0.0, 10.0), Shot(10.0, 20.0)]
        from cleancut.edl import EditDecisionList
        with patch("cleancut.classify_visual.make_ollama_client", return_value=_DeadClient()):
            with pytest.raises(RuntimeError, match="[Oo]llama"):
                scan_with_vlm(video, shots, [], EditDecisionList(), VLMParams(mode="all"))

    def test_flagged_categories_malformed_confidence(self):
        result = {"explicit": True, "confidence": "high"}
        assert _flagged_categories(result, VLMParams()) == []


class TestRunDetectorErrorPolicy:
    def test_catches_any_exception_not_just_runtime_error(self):
        from cleancut.pipeline import _run_detector

        def bad_detector():
            raise ValueError("cv2 exploded")

        assert _run_detector("Test detector", bad_detector) == []


class TestAudioEventsRobustness:
    def test_ffmpeg_failure_raises_runtime_error(self, tmp_path):
        """No audio track → ffmpeg fails → must be RuntimeError (skippable),
        not a raw CalledProcessError that kills the whole run."""
        from cleancut.audio_events import _extract_full_audio_to_wav

        def fail(cmd, **kwargs):
            raise subprocess.CalledProcessError(1, cmd)

        with patch("shutil.which", return_value="/usr/bin/ffmpeg"), \
             patch("subprocess.run", side_effect=fail):
            with pytest.raises(RuntimeError):
                _extract_full_audio_to_wav(tmp_path / "movie.mp4", None)

    def test_missing_librosa_raises_runtime_error_with_extra_hint(self, tmp_path, monkeypatch):
        from cleancut import audio_events

        monkeypatch.setitem(sys.modules, "librosa", None)
        with pytest.raises(RuntimeError, match="cleancut\\[audio\\]"):
            audio_events._load_audio(tmp_path / "x.wav")
