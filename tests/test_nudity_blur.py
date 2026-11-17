from __future__ import annotations

import shutil
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from cleancut.config import Config
from cleancut.edl import EditDecision, EditDecisionList
from cleancut.pipeline import PipelineOptions, build_edl, render
from cleancut.probe import audio_streams, probe_streams, probe_video_timing
from cleancut.scenes import Shot, detect_shots
from cleancut.sexual_activity import scan_sexual_activity


def test_scene_includes_contiguous_context_and_keeps_ambiguity_for_review(tmp_path, monkeypatch):
    source = tmp_path / "movie.mkv"
    source.touch()
    monkeypatch.setattr("cleancut.cache.CACHE_DIR", tmp_path / "cache")
    shots = [Shot(i, i + 1) for i in range(7)]
    with patch("cleancut.sexual_activity._shot_verdicts", return_value=[
        "absent", "continuation", "active", "continuation", "absent", "uncertain", "continuation",
    ]) as classify:
        result = scan_sexual_activity(source, shots, Config())
        assert scan_sexual_activity(source, shots, Config()) == result
    classify.assert_called_once()
    assert [(d.start, d.end, d.accepted) for d in result] == [(1, 4, True), (5, 6, False), (6, 7, False)]


def test_scan_mode_produces_nudity_blurs_and_keeps_sex_cut_boundaries(tmp_path):
    video = tmp_path / "movie.mkv"
    config = Config.load_defaults()
    config.blur_nudity_cut_sex = True
    config.analysis_proxy_enabled = False
    config.enabled_categories = {"sex", "nudity"}
    config.llm_enabled = config.vlm_enabled = True
    opts = PipelineOptions(video=video, burn_subs=False, use_visual=False, use_scenes=False)
    shots = [Shot(0, 6), Shot(6, 9)]

    def detect(_video, cfg):
        action = cfg.actions["nudity"]
        return EditDecisionList(decisions=[EditDecision(1, 2, action, "nudity", source="visual")])

    cut = EditDecision(6, 9, "cut", "sex", source="sexual-activity")
    with patch("cleancut.scenes.detect_shots", return_value=shots), \
         patch("cleancut.visual.scan_video", side_effect=detect), \
         patch("cleancut.sexual_activity.scan_sexual_activity", return_value=[cut]), \
         patch("cleancut.pipeline._get_subtitles_and_words") as dialogue:
        edl, _ = build_edl(opts, config)
    dialogue.assert_not_called()
    assert edl.by_action("cut") == [cut]
    assert edl.by_action("blur")[0].category == "nudity"


def test_activity_scan_uses_multiple_frames_from_a_single_shot(tmp_path, monkeypatch):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("Requires FFmpeg")
    pytest.importorskip("cv2")
    pytest.importorskip("scenedetect")
    video = tmp_path / "single-shot.mkv"
    subprocess.run([
        "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=black:s=96x64:r=10:d=2",
        "-c:v", "ffv1", str(video),
    ], check=True)
    monkeypatch.setattr("cleancut.cache.CACHE_DIR", tmp_path / "cache")
    shots = detect_shots(video, include_single_shot=True)
    assert shots == [Shot(0, 2)]
    with patch("cleancut.sexual_activity.make_ollama_client"), \
         patch("cleancut.sexual_activity._classify", return_value="absent") as classify:
        assert scan_sexual_activity(video, shots, Config()) == []
    classify.assert_called_once()
    assert len(classify.call_args.args[2]) == 3


def test_shot_detection_reads_variable_rate_timestamps_in_one_pass(tmp_path):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("Requires FFmpeg")
    cv2 = pytest.importorskip("cv2")
    pytest.importorskip("scenedetect")
    video = tmp_path / "variable-rate.mkv"
    subprocess.run([
        "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=black:s=96x64:r=10:d=4",
        "-vf", "drawbox=c=white:t=fill:enable='gte(n,20)',setpts='PTS+if(gte(N,20),5,0)'",
        "-fps_mode", "vfr", "-c:v", "ffv1", str(video),
    ], check=True)
    with patch.object(cv2, "VideoCapture", wraps=cv2.VideoCapture) as capture:
        shots = detect_shots(video, use_cache=False, include_single_shot=True)
    capture.assert_called_once()
    _, duration, video_end = probe_video_timing(video)
    assert shots == [Shot(0, 2.5), Shot(2.5, min(duration, video_end or duration))]


@pytest.mark.parametrize(("cut", "encoder", "audio"), [
    (False, "libx264", True), (True, "libx264", True), (False, "libx265", False), (True, "libx264", False),
])
def test_render_blurs_moving_regions_on_the_source_timeline(tmp_path, cut, encoder, audio):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("Requires FFmpeg")
    cv2 = pytest.importorskip("cv2")
    source, output = tmp_path / "source.mkv", tmp_path / "output.mp4"
    audio_args = ["-f", "lavfi", "-i", "sine=frequency=600:duration=3", "-c:a", "pcm_s16le"] if audio else []
    subprocess.run([
        "ffmpeg", "-v", "error", "-f", "lavfi", "-i",
        "nullsrc=s=96x64:r=10:d=3,geq=lum='if(mod(X+Y,2),32,220)':cb=128:cr=128",
        *audio_args, "-c:v", "ffv1", str(source),
    ], check=True)

    index = 0

    def detect_batch(images, batch_size):
        nonlocal index
        outputs = []
        for _ in images:
            box = [8 if index < 5 else 68, 22, 16, 16]
            outputs.append([] if index == 4 else [{"class": "BUTTOCKS_EXPOSED", "score": 0.8, "box": box}])
            index += 1
        return outputs

    detector = SimpleNamespace(detect_batch=detect_batch)
    edl = EditDecisionList(decisions=[EditDecision(1, 2, "blur", "nudity")])
    if cut:
        edl.add(EditDecision(0.2, 0.6, "cut", "sex"))
    opts = PipelineOptions(video=source, output=output, burn_subs=False)
    with patch("cleancut.blur.resolve_nudity_model"), \
         patch("cleancut.blur._open_detector", return_value=(cv2, detector)):
        render(edl, [], opts, Config(encoder=encoder, render_validation="full"))
    assert index == 10
    assert bool(audio_streams(probe_streams(output))) == audio

    cap = cv2.VideoCapture(str(output))
    frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()

    def deviation(frame_index, x):
        return frames[frame_index][24:34, x:x + 10].std()

    shift = 4 if cut else 0
    assert len(frames) == 30 - shift
    assert deviation(9 - shift, 10) > 50
    assert deviation(11 - shift, 10) < 8 and deviation(11 - shift, 70) > 50
    assert deviation(14 - shift, 10) < 8 and deviation(14 - shift, 70) < 8
    assert deviation(17 - shift, 10) > 50 and deviation(17 - shift, 70) < 8
    assert deviation(20 - shift, 70) > 50
