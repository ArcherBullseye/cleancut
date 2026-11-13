"""Tests for the local two-pass NudeNet scanner."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from cleancut.config import Config
from cleancut.edl import EditDecisionList
from cleancut.nudity_model import NudityModel
from cleancut.scenes import Shot
from cleancut.visual import (
    NudityEvidence,
    _confirmed_edl,
    _dense_times,
    _infer_samples,
    _is_explicit,
    scan_video,
)

HIT = [{"class": "FEMALE_BREAST_EXPOSED", "score": 0.90}]
MEDIUM_HIT = [{"class": "FEMALE_BREAST_EXPOSED", "score": 0.60}]
MODEL = NudityModel("test-640m", Path("/tmp/test.onnx"), 640)


def _config(**values) -> Config:
    config = Config.load_defaults()
    config.visual_threshold = 0.45
    config.nudity_strong_threshold = 0.90
    config.nudity_min_confirmations = 3
    config.nudity_temporal_padding_seconds = 0.5
    config.nudity_coreml = False
    for key, value in values.items():
        setattr(config, key, value)
    return config


class FakeCapture:
    def __init__(self, fps=10.0, frames=100):
        self.fps = fps
        self.frames = frames

    def isOpened(self):
        return True

    def get(self, prop):
        return self.fps if prop == 5 else self.frames

    def grab(self):
        return True

    def read(self):
        return True, object()

    def release(self):
        pass


CV2 = SimpleNamespace(CAP_PROP_FPS=5, CAP_PROP_FRAME_COUNT=7)


def test_is_explicit_uses_only_exposed_classes_above_threshold():
    assert _is_explicit(HIT, 0.7)
    assert not _is_explicit([{"class": "FACE_FEMALE", "score": 0.99}], 0.1)
    assert not _is_explicit([{"class": "FEMALE_BREAST_EXPOSED", "score": 0.4}], 0.7)


def test_dense_rescan_surrounds_candidate_at_requested_rate():
    config = _config(nudity_rescan_fps=4, nudity_rescan_window_seconds=1)
    samples = _dense_times([5.0], 10.0, config)
    times = [time for time, _ in samples]
    assert 4.0 in times
    assert 5.0 in times
    assert 6.0 in times
    assert 4.25 in times


def test_batch_inference_is_used():
    detector = MagicMock()
    detector.detect_batch.side_effect = lambda frames, batch_size: [[] for _ in frames]
    cap = FakeCapture(fps=10, frames=50)
    result = _infer_samples(
        detector, CV2, cap, 10, [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)],
        batch_size=2, description="test",
    )
    assert len(result) == 3
    assert detector.detect_batch.call_count == 2
    assert detector.detect.call_count == 0


def test_single_strong_hit_is_accepted():
    detector = SimpleNamespace(_cleancut_model="640m", _cleancut_backend="CPU")
    edl = _confirmed_edl({5.0: HIT}, 20.0, _config(), "cut", detector)
    assert len(edl.decisions) == 1
    assert edl.decisions[0].start == 4.5
    assert edl.decisions[0].source == "visual"


def test_two_nearby_medium_hits_are_review_only():
    detector = SimpleNamespace()
    edl = _confirmed_edl(
        {5.0: MEDIUM_HIT, 5.2: MEDIUM_HIT}, 20.0, _config(), "cut", detector
    )
    assert len(edl.decisions) == 1
    assert not edl.decisions[0].accepted
    assert "2 samples" in edl.decisions[0].reason


def test_isolated_medium_hit_is_rejected():
    edl = _confirmed_edl({5.0: MEDIUM_HIT}, 20.0, _config(), "cut", object())
    assert not edl.decisions


def test_brief_hit_in_long_shot_is_not_diluted_by_shot_fraction(tmp_path):
    video = tmp_path / "movie.mp4"
    video.write_bytes(b"video")
    detector = SimpleNamespace(_cleancut_model="640m", _cleancut_backend="CPU")
    coarse = [(50.0, HIT)]
    dense = [(49.9, HIT), (50.0, HIT)]
    with patch("cleancut.visual.resolve_nudity_model", return_value=MODEL), \
         patch("cleancut.visual._open_detector", return_value=(CV2, detector)), \
         patch("cleancut.visual._open_capture", side_effect=[FakeCapture(frames=1000), FakeCapture(frames=1000)]), \
         patch("cleancut.visual._infer_samples", side_effect=[coarse, dense]):
        edl = scan_video(
            video, _config(), shots=[Shot(start=0.0, end=100.0)], use_cache=False
        )
    assert len(edl.decisions) == 1
    assert 49.0 < edl.decisions[0].start < 50.0
    assert edl.decisions[0].end < 51.0


def test_no_candidates_returns_empty_edl(tmp_path):
    video = tmp_path / "movie.mp4"
    video.write_bytes(b"video")
    detector = SimpleNamespace(_cleancut_model="640m", _cleancut_backend="CPU")
    with patch("cleancut.visual.resolve_nudity_model", return_value=MODEL), \
         patch("cleancut.visual._open_detector", return_value=(CV2, detector)), \
         patch("cleancut.visual._open_capture", return_value=FakeCapture()), \
         patch("cleancut.visual._infer_samples", return_value=[]):
        result = scan_video(video, _config(), use_cache=False)
    assert isinstance(result, EditDecisionList)
    assert not result.decisions


def test_all_frames_finds_a_flash_and_uses_its_source_timestamps(tmp_path, monkeypatch):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("Requires FFmpeg")
    cv2 = pytest.importorskip("cv2")
    video = tmp_path / "flash.mkv"
    # Put white on frame 1. Its PTS is .04s, and the following frame is at .24s.
    # A half-second sample would miss it, and frame index / average FPS is wrong.
    subprocess.run([
        "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=black:s=32x32:r=25:d=0.48",
        "-vf", "drawbox=c=white:t=fill:enable='eq(n,1)',setpts='PTS+if(gte(N,2),4,0)'",
        "-fps_mode", "vfr", "-c:v", "ffv1", str(video),
    ], check=True)
    monkeypatch.setattr("cleancut.cache.CACHE_DIR", tmp_path / "cache")

    batch_sizes = []

    def detect_batch(frames, batch_size):
        batch_sizes.append(batch_size)
        return [HIT if frame.mean() > 200 else [] for frame in frames]

    detector = SimpleNamespace(detect_batch=detect_batch)
    config = _config(visual_sample_seconds=0.5, nudity_temporal_padding_seconds=0)
    with patch("cleancut.visual.resolve_nudity_model", return_value=MODEL), \
         patch("cleancut.visual._open_detector", return_value=(cv2, detector)):
        assert not scan_video(video, config).decisions
        batch_sizes.clear()
        config.nudity_all_frames = True
        evidence: list[NudityEvidence] = []
        edl = scan_video(video, config, evidence=evidence)
        assert len(evidence) == 1 and evidence[0].frame == 1
        cached_evidence: list[NudityEvidence] = []
        assert scan_video(video, config, evidence=cached_evidence) == edl
        assert cached_evidence == evidence
    assert sum(batch_sizes) == 12
    cut, = edl.by_action("cut")
    assert (cut.start, cut.end) == pytest.approx((0.04, 0.24))


def test_all_frames_requires_temporally_separated_confirmation():
    config = _config(nudity_all_frames=True)
    hit = [{"class": "FEMALE_BREAST_EXPOSED", "score": 0.8}]
    observations = {n / 60: hit for n in range(3)}
    weak = _confirmed_edl(observations, 10, config, "cut", object())
    assert len(weak.decisions) == 1 and not weak.decisions[0].accepted
    observations.update({0.1: hit, 0.2: hit})
    assert len(_confirmed_edl(observations, 10, config, "cut", object()).by_action("cut")) == 1


def test_cache_hit_skips_detector(tmp_path):
    video = tmp_path / "movie.mp4"
    video.write_bytes(b"video")
    cached = {"decisions": [{
        "start": 5.0, "end": 6.0, "action": "cut", "category": "nudity",
        "reason": "cached", "source": "visual", "text_before": "",
        "text_after": "", "accepted": True,
    }]}
    with patch("cleancut.visual.resolve_nudity_model", return_value=MODEL), \
         patch("cleancut.cache.config_hash", return_value="hash"), \
         patch("cleancut.cache.load", return_value=cached), \
         patch("cleancut.visual._open_detector") as opened:
        result = scan_video(video, _config(), use_cache=True)
    assert result.decisions[0].start == 5.0
    opened.assert_not_called()


def test_missing_visual_dependencies_has_clear_error():
    from cleancut.visual import _open_detector

    with patch.dict(sys.modules, {"cv2": None, "nudenet": None}), \
         pytest.raises(RuntimeError, match="Visual detection requires"):
        _open_detector(_config(), MODEL)


def test_coreml_session_failure_keeps_cpu_detector():
    from cleancut.visual import _open_detector

    original_session = object()
    detector = SimpleNamespace(onnx_session=original_session, input_name="images")
    nudenet = SimpleNamespace(NudeDetector=MagicMock(return_value=detector))
    ort = SimpleNamespace(
        get_available_providers=lambda: ["CoreMLExecutionProvider", "CPUExecutionProvider"],
        InferenceSession=MagicMock(side_effect=RuntimeError("compile failed")),
    )
    config = _config(nudity_coreml=True)
    with patch.dict(sys.modules, {"cv2": CV2, "nudenet": nudenet, "onnxruntime": ort}), \
         patch("cleancut.visual.platform.system", return_value="Darwin"):
        _, result = _open_detector(config, MODEL)
    assert result.onnx_session is original_session
    assert result._cleancut_backend == "CPU"
