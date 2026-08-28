"""Tests for editor.py helpers not covered by editor_math tests."""

from cleancut.editor import Range, _video_encoder_args


def test_video_encoder_libx264_args():
    args = _video_encoder_args("libx264", 18)
    assert "libx264" in args
    assert "-crf" in args
    assert "18" in args
    assert "-preset" in args


def test_video_encoder_videotoolbox_args():
    args = _video_encoder_args("videotoolbox", 20)
    assert "h264_videotoolbox" in args
    assert "-q:v" in args
    # Quality should be mapped to a q-value
    q_idx = args.index("-q:v") + 1
    q = int(args[q_idx])
    assert 30 <= q <= 100


def test_video_encoder_higher_quality_higher_q_for_videotoolbox():
    # Lower CRF (higher visual quality) should map to higher VT q-value
    args_high = _video_encoder_args("videotoolbox", 16)
    args_low = _video_encoder_args("videotoolbox", 26)
    q_high = int(args_high[args_high.index("-q:v") + 1])
    q_low = int(args_low[args_low.index("-q:v") + 1])
    assert q_high > q_low


def test_range_duration():
    r = Range(10.0, 25.5)
    assert r.duration == 15.5


def test_range_negative_duration_clamped():
    r = Range(20.0, 10.0)
    assert r.duration == 0.0
