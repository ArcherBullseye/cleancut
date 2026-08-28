import numpy as np

from cleancut.audio_events import _slice_audio


def test_slice_audio_centered():
    sr = 16000
    audio = np.arange(sr * 10, dtype=np.float32)  # 10s ramp
    clip = _slice_audio(audio, sr, center_seconds=5.0, clip_seconds=4.0)
    assert len(clip) == sr * 4
    # Center should be ~5s into the audio = index 5*sr in original
    # clip starts at 3s = index 3*sr
    assert int(clip[0]) == 3 * sr


def test_slice_audio_at_start_pads():
    sr = 16000
    audio = np.arange(sr * 10, dtype=np.float32)
    clip = _slice_audio(audio, sr, center_seconds=0.5, clip_seconds=4.0)
    # Should still be exactly 4 seconds even though we requested before-zero
    assert len(clip) == sr * 4
    # First sample should be at t=0 (start clamped)
    assert int(clip[0]) == 0


def test_slice_audio_at_end_pads():
    sr = 16000
    audio = np.arange(sr * 10, dtype=np.float32)
    clip = _slice_audio(audio, sr, center_seconds=9.5, clip_seconds=4.0)
    assert len(clip) == sr * 4
    # Trailing zeros from padding
    assert clip[-1] == 0.0
