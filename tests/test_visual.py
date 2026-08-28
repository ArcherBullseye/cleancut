"""Tests for cleancut/visual.py — NudeNet-based visual scanning.

All cv2, NudeNet, and cache calls are mocked so no GPU/model is needed.
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from cleancut.config import Config
from cleancut.edl import EditDecisionList
from cleancut.visual import _is_explicit


# ---------------------------------------------------------------------------
# _is_explicit helper
# ---------------------------------------------------------------------------

class TestIsExplicit:
    def test_returns_true_for_explicit_class_above_threshold(self):
        detections = [{"class": "FEMALE_BREAST_EXPOSED", "score": 0.85}]
        assert _is_explicit(detections, threshold=0.7) is True

    def test_returns_false_for_class_below_threshold(self):
        detections = [{"class": "FEMALE_BREAST_EXPOSED", "score": 0.50}]
        assert _is_explicit(detections, threshold=0.7) is False

    def test_returns_false_for_non_explicit_class(self):
        detections = [{"class": "FACE_FEMALE", "score": 0.99}]
        assert _is_explicit(detections, threshold=0.1) is False

    def test_returns_false_for_empty_detections(self):
        assert _is_explicit([], threshold=0.5) is False

    def test_any_explicit_class_triggers_true(self):
        detections = [
            {"class": "FACE_FEMALE", "score": 0.99},
            {"class": "MALE_GENITALIA_EXPOSED", "score": 0.80},
        ]
        assert _is_explicit(detections, threshold=0.75) is True


# ---------------------------------------------------------------------------
# scan_video with mocked cv2 + NudeDetector
# ---------------------------------------------------------------------------

def _make_config(**kw) -> Config:
    cfg = Config.load_defaults()
    cfg.visual_threshold = 0.7
    cfg.visual_sample_seconds = 1.0
    cfg.visual_min_streak = 2
    cfg.visual_shot_hit_fraction = 0.5
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def _build_cv2_mock(fps: float = 24.0, total_frames: int = 120,
                    frame_ok: bool = True):
    """Return a mock cv2 module with a VideoCapture that reads frames."""
    mock_cv2 = MagicMock()
    mock_cv2.CAP_PROP_FPS = 5
    mock_cv2.CAP_PROP_FRAME_COUNT = 7

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.side_effect = lambda prop: fps if prop == 5 else total_frames
    mock_cap.read.return_value = (frame_ok, MagicMock() if frame_ok else None)
    mock_cv2.VideoCapture.return_value = mock_cap
    return mock_cv2, mock_cap


class TestScanVideoStreakMode:
    """scan_video in streak mode (no shots argument)."""

    def test_returns_edl_instance(self, tmp_path):
        from cleancut.visual import scan_video
        fake_video = tmp_path / "v.mp4"
        fake_video.write_bytes(b"\x00")
        config = _make_config(visual_sample_seconds=1.0, visual_min_streak=2)

        mock_cv2, _ = _build_cv2_mock(fps=24.0, total_frames=72)  # 3 s
        mock_detector = MagicMock()
        mock_detector.detect.return_value = []  # nothing detected

        mock_nudenet = MagicMock()
        mock_nudenet.NudeDetector.return_value = mock_detector

        with patch.dict(sys.modules, {"cv2": mock_cv2, "nudenet": mock_nudenet}), \
             patch("cleancut.cache.config_hash", return_value="h"), \
             patch("cleancut.cache.load", return_value=None), \
             patch("cleancut.cache.save"):
            result = scan_video(fake_video, config, shots=None, use_cache=False)

        assert isinstance(result, EditDecisionList)

    def test_no_detections_returns_empty_edl(self, tmp_path):
        from cleancut.visual import scan_video
        fake_video = tmp_path / "v.mp4"
        fake_video.write_bytes(b"\x00")
        config = _make_config(visual_sample_seconds=1.0, visual_min_streak=1)

        mock_cv2, _ = _build_cv2_mock(fps=24.0, total_frames=48)
        mock_detector = MagicMock()
        mock_detector.detect.return_value = []

        mock_nudenet = MagicMock()
        mock_nudenet.NudeDetector.return_value = mock_detector

        with patch.dict(sys.modules, {"cv2": mock_cv2, "nudenet": mock_nudenet}), \
             patch("cleancut.cache.config_hash", return_value="h"), \
             patch("cleancut.cache.load", return_value=None), \
             patch("cleancut.cache.save"):
            result = scan_video(fake_video, config, shots=None, use_cache=False)

        assert len(result.decisions) == 0

    def test_streak_of_hits_produces_cut(self, tmp_path):
        """Three consecutive frames all flagged should produce a cut decision."""
        from cleancut.visual import scan_video
        fake_video = tmp_path / "v.mp4"
        fake_video.write_bytes(b"\x00")
        # sample_seconds=1.0, min_streak=2, 5 second video
        config = _make_config(visual_sample_seconds=1.0, visual_min_streak=2)

        mock_cv2, mock_cap = _build_cv2_mock(fps=10.0, total_frames=50)  # 5 seconds
        mock_detector = MagicMock()
        # All frames flag nudity above threshold
        mock_detector.detect.return_value = [
            {"class": "FEMALE_BREAST_EXPOSED", "score": 0.90}
        ]

        mock_nudenet = MagicMock()
        mock_nudenet.NudeDetector.return_value = mock_detector

        with patch.dict(sys.modules, {"cv2": mock_cv2, "nudenet": mock_nudenet}), \
             patch("cleancut.cache.config_hash", return_value="h"), \
             patch("cleancut.cache.load", return_value=None), \
             patch("cleancut.cache.save"):
            result = scan_video(fake_video, config, shots=None, use_cache=False)

        assert len(result.decisions) >= 1
        d = result.decisions[0]
        assert d.action == config.actions.get("nudity", "cut")
        assert d.category == "nudity"
        assert d.source == "visual"

    def test_single_hit_below_streak_not_emitted(self, tmp_path):
        """A single flagged frame with min_streak=3 should NOT emit a cut."""
        from cleancut.visual import scan_video
        fake_video = tmp_path / "v.mp4"
        fake_video.write_bytes(b"\x00")
        config = _make_config(visual_sample_seconds=1.0, visual_min_streak=3)

        # 5 second video: hit on frame 2 only, everything else clean
        mock_cv2, mock_cap = _build_cv2_mock(fps=10.0, total_frames=50)
        hit_detection = [{"class": "FEMALE_BREAST_EXPOSED", "score": 0.90}]
        clean = []
        call_results = [clean, hit_detection, clean, clean, clean]
        mock_detector = MagicMock()
        mock_detector.detect.side_effect = call_results + [clean] * 100

        mock_nudenet = MagicMock()
        mock_nudenet.NudeDetector.return_value = mock_detector

        with patch.dict(sys.modules, {"cv2": mock_cv2, "nudenet": mock_nudenet}), \
             patch("cleancut.cache.config_hash", return_value="h"), \
             patch("cleancut.cache.load", return_value=None), \
             patch("cleancut.cache.save"):
            result = scan_video(fake_video, config, shots=None, use_cache=False)

        assert len(result.decisions) == 0

    def test_cache_hit_returns_without_calling_cv2(self, tmp_path):
        """When the cache returns a hit, cv2 should never be imported/called."""
        from cleancut.visual import scan_video
        fake_video = tmp_path / "v.mp4"
        fake_video.write_bytes(b"\x00")
        config = _make_config()

        cached_data = {
            "decisions": [
                {
                    "start": 5.0, "end": 10.0,
                    "action": "cut", "category": "nudity",
                    "reason": "cached", "source": "visual",
                    "text_before": "", "text_after": "",
                    "accepted": True,
                }
            ]
        }

        with patch("cleancut.cache.config_hash", return_value="h"), \
             patch("cleancut.cache.load", return_value=cached_data) as mock_load:
            result = scan_video(fake_video, config, shots=None, use_cache=True)

        assert len(result.decisions) == 1
        assert result.decisions[0].start == 5.0
        mock_load.assert_called_once()

    def test_raises_runtime_error_when_nudenet_not_installed(self, tmp_path):
        from cleancut.visual import scan_video
        fake_video = tmp_path / "v.mp4"
        fake_video.write_bytes(b"\x00")
        config = _make_config()

        with patch("cleancut.cache.config_hash", return_value="h"), \
             patch("cleancut.cache.load", return_value=None), \
             patch.dict(sys.modules, {"cv2": None, "nudenet": None}):
            with pytest.raises(RuntimeError, match="visual"):
                scan_video(fake_video, config, shots=None, use_cache=False)


class TestScanVideoShotAwareMode:
    """scan_video in shot-aware mode (shots list supplied)."""

    def test_shot_above_hit_fraction_produces_cut(self, tmp_path):
        from cleancut.visual import scan_video
        from cleancut.scenes import Shot
        fake_video = tmp_path / "v.mp4"
        fake_video.write_bytes(b"\x00")
        # hit_fraction=0.5: if ≥50% of frames flag → cut
        config = _make_config(visual_shot_hit_fraction=0.5, visual_sample_seconds=1.0)
        shots = [Shot(start=0.0, end=6.0)]

        mock_cv2, mock_cap = _build_cv2_mock(fps=10.0, total_frames=60)
        # All frames flag nudity
        mock_detector = MagicMock()
        mock_detector.detect.return_value = [
            {"class": "FEMALE_BREAST_EXPOSED", "score": 0.90}
        ]

        mock_nudenet = MagicMock()
        mock_nudenet.NudeDetector.return_value = mock_detector

        with patch.dict(sys.modules, {"cv2": mock_cv2, "nudenet": mock_nudenet}), \
             patch("cleancut.cache.config_hash", return_value="h"), \
             patch("cleancut.cache.load", return_value=None), \
             patch("cleancut.cache.save"):
            result = scan_video(fake_video, config, shots=shots, use_cache=False)

        assert len(result.decisions) >= 1
        d = result.decisions[0]
        assert d.start == shots[0].start
        assert d.end == shots[0].end
        assert d.source == "visual-shot"

    def test_shot_below_hit_fraction_not_cut(self, tmp_path):
        from cleancut.visual import scan_video
        from cleancut.scenes import Shot
        fake_video = tmp_path / "v.mp4"
        fake_video.write_bytes(b"\x00")
        config = _make_config(visual_shot_hit_fraction=0.8, visual_sample_seconds=1.0)
        shots = [Shot(start=0.0, end=6.0)]

        mock_cv2, mock_cap = _build_cv2_mock(fps=10.0, total_frames=60)
        # Only every other frame flags nudity → ~50% hit rate, below 0.8
        call_count = [0]

        def alternating_detect(frame):
            call_count[0] += 1
            if call_count[0] % 2 == 0:
                return [{"class": "FEMALE_BREAST_EXPOSED", "score": 0.90}]
            return []

        mock_detector = MagicMock()
        mock_detector.detect.side_effect = alternating_detect

        mock_nudenet = MagicMock()
        mock_nudenet.NudeDetector.return_value = mock_detector

        with patch.dict(sys.modules, {"cv2": mock_cv2, "nudenet": mock_nudenet}), \
             patch("cleancut.cache.config_hash", return_value="h"), \
             patch("cleancut.cache.load", return_value=None), \
             patch("cleancut.cache.save"):
            result = scan_video(fake_video, config, shots=shots, use_cache=False)

        assert len(result.decisions) == 0
