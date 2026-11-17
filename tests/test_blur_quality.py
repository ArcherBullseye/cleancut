from __future__ import annotations

import shutil
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from cleancut.blur import prepare_blur
from cleancut.config import Config
from cleancut.editor import apply_mutes_and_subs
from cleancut.editor_ranges import Range


@pytest.mark.parametrize("localize", [False, True])
def test_blur_covers_a_frame_held_through_a_range_between_timestamps(tmp_path, localize):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("Requires FFmpeg")
    cv2 = pytest.importorskip("cv2")
    source, output = tmp_path / "hold.mkv", tmp_path / "blurred.mp4"
    subprocess.run([
        "ffmpeg", "-v", "error", "-f", "lavfi", "-i",
        "nullsrc=s=96x64:r=5:d=2,geq=lum='if(mod(X+Y,2),32,220)':cb=128:cr=128",
        "-vf", "select='eq(n,0)+eq(n,1)+eq(n,8)+eq(n,9)'", "-fps_mode", "vfr", "-c:v", "ffv1", str(source),
    ], check=True)

    # The frame at 0.2s remains visible until 1.6s. No frame starts in this range.
    hit = {"class": "BUTTOCKS_EXPOSED", "score": 0.8, "box": [10, 12, 20, 24]}
    detector = SimpleNamespace(detect_batch=lambda frames, batch_size: [[hit] if localize else [] for _ in frames])
    span = Range(0.25, 0.35) if localize else Range(0.65, 0.95)
    with patch("cleancut.blur.resolve_nudity_model"), \
         patch("cleancut.blur._open_detector", return_value=(cv2, detector)):
        blur = prepare_blur(source, [span], Config(), tmp_path / "blur.commands")
    apply_mutes_and_subs(source, [], None, output, burn_subs=False, validation="full", blur=blur)

    capture = cv2.VideoCapture(str(output))
    frames = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        capture.release()
    assert frames[0].std() > 50
    held_frame = frames[len(frames) // 2]
    assert held_frame[20:28, 14:24].std() < 8
    detail = held_frame[20:28, 70:80].std()
    assert detail > 50 if localize else detail < 8
    assert frames[-1].std() > 50
