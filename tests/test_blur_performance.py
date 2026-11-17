from __future__ import annotations

import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from cleancut import cache
from cleancut.blur import prepare_blur
from cleancut.config import Config
from cleancut.editor_ranges import Range
from cleancut.nudity_model import NudityModel


def test_rerender_reuses_blur_commands_for_unchanged_ranges(tmp_path, monkeypatch):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("Requires FFmpeg")
    cv2 = pytest.importorskip("cv2")
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "cache")
    source = tmp_path / "source.mkv"
    subprocess.run([
        "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=5:duration=2",
        "-c:v", "ffv1", str(source),
    ], check=True)
    weights = tmp_path / "local.onnx"
    weights.touch()
    model = NudityModel("local", weights, 320)
    hit = {"class": "BUTTOCKS_EXPOSED", "score": 0.8, "box": [10, 12, 20, 24]}
    detector = SimpleNamespace(detect_batch=lambda frames, batch_size: [[hit] for _ in frames])
    config = Config()
    rerender_cmd = tmp_path / "rerender.commands"
    with patch("cleancut.blur.resolve_nudity_model", return_value=model), \
         patch("cleancut.blur._open_detector", return_value=(cv2, detector)) as load_detector:
        fresh = prepare_blur(source, [Range(0.4, 1.0)], config, tmp_path / "first.commands")
        with patch("cleancut.blur._open_capture") as capture, patch("cleancut.blur.probe_video_timing") as probe:
            reused = prepare_blur(source, [Range(0.4, 1.0)], config, rerender_cmd)
        assert reused.commands.read_bytes() == fresh.commands.read_bytes()
        assert reused.graph("[out]", "libx264") == fresh.graph("[out]", "libx264")
        capture.assert_not_called()
        probe.assert_not_called()
        load_detector.assert_called_once()

        prepare_blur(source, [Range(0.4, 1.1)], config, rerender_cmd)
        assert load_detector.call_count == 2
        # A same-size replacement or modification within the same second must invalidate the mask.
        status = source.stat()
        mtime_ns = status.st_mtime_ns - (1 if status.st_mtime_ns % 1_000_000_000 else -1)
        os.utime(source, ns=(status.st_atime_ns, mtime_ns))
        prepare_blur(source, [Range(0.4, 1.1)], config, rerender_cmd)
        assert load_detector.call_count == 3


def test_concurrent_cache_saves_publish_complete_results(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "cache")
    source = tmp_path / "source.mkv"
    source.touch()
    payloads = [{"commands": f"mask {i}\n" * 5000} for i in range(8)]
    with ThreadPoolExecutor(max_workers=8) as workers:
        paths = list(workers.map(lambda data: cache.save(source, "blur", "key", data), payloads))
    assert cache.load(source, "blur", "key") in payloads
    assert list(cache.CACHE_DIR.iterdir()) == paths[:1]
