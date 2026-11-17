from __future__ import annotations

import shutil
import subprocess
from unittest.mock import patch

import pytest

from cleancut.config import Config
from cleancut.scenes import Shot
from cleancut.sexual_activity import scan_sexual_activity


@pytest.mark.parametrize(("duration", "bright_frame"), [(0.6, 3), (0.6, 5), (1.4, 13)])
def test_activity_scan_checks_short_shot_middle_and_unsampled_tail(tmp_path, monkeypatch, duration, bright_frame):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("Requires FFmpeg")
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    video = tmp_path / "shot.mkv"
    subprocess.run([
        "ffmpeg", "-v", "error", "-f", "lavfi", "-i", f"color=black:s=64x64:r=10:d={duration}",
        "-vf", f"drawbox=color=white:t=fill:enable='eq(n,{bright_frame})'", "-c:v", "ffv1", str(video),
    ], check=True)
    monkeypatch.setattr("cleancut.cache.CACHE_DIR", tmp_path / "cache")

    def classify(_client, _model, images, _instruction, **_kwargs):
        for image in images:
            frame = cv2.imdecode(np.frombuffer(image, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if frame.mean() > 225:
                return "active"
        return "absent"

    with patch("cleancut.sexual_activity.make_ollama_client"), \
         patch("cleancut.sexual_activity._classify", side_effect=classify) as detect:
        result = scan_sexual_activity(video, [Shot(0, duration)], Config())

    detect.assert_called_once()
    assert [(decision.start, decision.end, decision.accepted) for decision in result] == [(0, duration, True)]
