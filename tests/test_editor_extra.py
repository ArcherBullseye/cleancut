"""Tests for editor.py helpers not covered by editor_math tests."""

import pytest

from cleancut.editor import (
    Range,
    _can_stream_copy_to_apple_mp4,
    _video_encoder_args,
    videotoolbox_quality,
)


@pytest.mark.parametrize("encoder", ["videotoolbox", "hevc_videotoolbox"])
@pytest.mark.parametrize("quality,expected", [(14,92), (18,84), (20,80), (23,74), (32,56)])
def test_hardware_quality_first_mapping(encoder, quality, expected):
    args = _video_encoder_args(encoder, quality)
    assert args[args.index("-q:v") + 1] == str(expected)
    assert args[args.index("-b:v") + 1] == "0"


def test_hardware_quality_is_bounded_and_monotonic():
    values = [videotoolbox_quality(q) for q in range(-10, 70)]
    assert all(30 <= q <= 95 for q in values)
    assert values == sorted(values, reverse=True)


@pytest.mark.parametrize("encoder", ["libx264", "libx265"])
def test_software_crf_not_remapped(encoder):
    args = _video_encoder_args(encoder, 18)
    assert args[args.index("-crf") + 1] == "18"
    assert "-q:v" not in args


def test_video_encoder_libx264_args():
    args = _video_encoder_args("libx264", 18)
    assert "libx264" in args
    assert "-crf" in args
    assert "18" in args
    assert "-preset" in args
    assert args[args.index("-pix_fmt") + 1] == "yuv420p"


def test_video_encoder_videotoolbox_args():
    args = _video_encoder_args("videotoolbox", 20)
    assert "h264_videotoolbox" in args
    assert "-q:v" in args
    # Quality should be mapped to a q-value
    q_idx = args.index("-q:v") + 1
    q = int(args[q_idx])
    assert 30 <= q <= 100
    assert args[args.index("-pix_fmt") + 1] == "yuv420p"


def test_video_encoder_higher_quality_higher_q_for_videotoolbox():
    # Lower CRF (higher visual quality) should map to higher VT q-value
    args_high = _video_encoder_args("videotoolbox", 16)
    args_low = _video_encoder_args("videotoolbox", 26)
    q_high = int(args_high[args_high.index("-q:v") + 1])
    q_low = int(args_low[args_low.index("-q:v") + 1])
    assert q_high > q_low


def test_apple_mp4_stream_copy_compatibility():
    assert _can_stream_copy_to_apple_mp4("h264", "yuv420p")
    assert _can_stream_copy_to_apple_mp4("hevc", "yuv420p10le")
    assert not _can_stream_copy_to_apple_mp4("h264", "yuv444p")
    assert not _can_stream_copy_to_apple_mp4("vp9", "yuv420p")


def test_range_duration():
    r = Range(10.0, 25.5)
    assert r.duration == 15.5


def test_range_negative_duration_clamped():
    r = Range(20.0, 10.0)
    assert r.duration == 0.0
