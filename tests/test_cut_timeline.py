"""Regression coverage for source -> edited timing, including real media."""
import shutil
import subprocess
from fractions import Fraction
from unittest.mock import patch

import numpy as np
import pytest

from cleancut.editor import Range, apply_cuts, shift_ranges_after_cuts
from cleancut.editor_ranges import adjust_subtitles_for_cuts, normalize_cuts, shift_after_cuts
from cleancut.subtitles import Subtitle


def test_mute_starting_inside_cut_starts_at_join_not_movie_start():
    shifted = shift_ranges_after_cuts([Range(12, 23)], [Range(10, 20)])
    assert [(r.start, r.end) for r in shifted] == [(10, 13)]


@pytest.mark.parametrize("interval, expected", [
    ((5, 12), (5, 10)),      # tail removed
    ((10, 20), None),        # completely removed, including exact boundaries
    ((20, 23), (10, 13)),    # cut end is the first surviving instant
    ((12, 45), (10, 30)),    # endpoints in DIFFERENT cuts, middle survives
    ((5, 55), (5, 35)),      # spans multiple cuts
    ((22, 25), (12, 15)),
])
def test_intervals_and_subtitles_use_same_cut_map(interval, expected):
    cuts = [Range(40, 50), Range(15, 20), Range(10, 17)]  # unsorted overlaps
    shifted = shift_ranges_after_cuts([Range(*interval)], cuts)
    subs = adjust_subtitles_for_cuts([Subtitle(1, *interval, "dialogue")], cuts)
    assert [(r.start, r.end) for r in shifted] == ([] if expected is None else [expected])
    assert [(s.start, s.end) for s in subs] == ([] if expected is None else [expected])


def test_cut_plan_clips_invalid_bounds_and_unions_removed_time():
    cuts = normalize_cuts([Range(-5, 2), Range(1, 3), Range(8, 14),
                           Range(15, 16), Range(6, 4), Range(7, 7)], duration=10)
    assert [(r.start, r.end) for r in cuts] == [(0, 3), (8, 10)]
    assert shift_after_cuts(3, cuts) == 0
    assert shift_after_cuts(7, cuts) == 4
    assert shift_after_cuts(10, cuts) == 5
    with pytest.raises(ValueError, match="finite"):
        normalize_cuts([Range(float("nan"), 2)])


def test_pipeline_passes_identical_cut_plan_to_speech_and_video(tmp_path):
    from cleancut.config import Config
    from cleancut.edl import EditDecision, EditDecisionList
    from cleancut.pipeline import PipelineOptions, render
    from cleancut.probe import Stream

    edl = EditDecisionList(decisions=[
        EditDecision(-5, 2, "cut", "nudity"),
        EditDecision(1, 3, "cut", "nudity"),
        EditDecision(8, 20, "cut", "nudity"),
        EditDecision(15, 16, "cut", "violence", accepted=False),
        EditDecision(3, 4, "mute", "profanity"),
    ])
    with patch("cleancut.pipeline.probe_duration", return_value=10), \
         patch("cleancut.probe.probe_streams", return_value=[Stream(1, "aac", "audio")]), \
         patch("cleancut.speech.prepare_replacements", return_value=[]) as speech, \
         patch("cleancut.pipeline.apply_cuts") as video, \
         patch("cleancut.pipeline.apply_mutes_and_subs") as mix:
        render(edl, [], PipelineOptions(video=tmp_path / "source", output=tmp_path / "out"),
               Config(profanity_audio="replace", encoder="libx264"))
    assert speech.call_args.kwargs["cuts"] == video.call_args.args[1] == [Range(0, 3), Range(8, 10)]
    assert mix.call_args.kwargs["mutes"] == [Range(0, 1)]
    # Rendering never rewrites the source-timeline EDL or shifts later cuts twice.
    assert edl.decisions[1].start == 1


def audio_samples(path):
    data = subprocess.check_output([
        "ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0",
        "-ac", "1", "-ar", "24000", "-f", "s16le", "-",
    ])
    return np.frombuffer(data, dtype="<i2").astype(float) / 32768


@pytest.mark.parametrize("rate", ["24", "24000/1001", "30"])
def test_repeated_fractional_cuts_do_not_shift_late_audio(tmp_path, rate):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("Requires real FFmpeg")
    source, out = tmp_path / "source.mkv", tmp_path / "cut.mp4"
    # 24 fractional cuts accumulate concat's video-frame padding. A short
    # audible marker near the end must still land on the mathematical timeline.
    subprocess.run([
        "ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
        (f"color=c=black:s=64x64:r={rate}:d=30,"
         "drawbox=c=white:t=fill:enable='between(t,28,28.4)'"), "-f", "lavfi", "-i",
        "aevalsrc=0.3*sin(2*PI*440*t)*between(t\\,28\\,28.4):s=24000:d=30",
        "-c:v", "libx264", "-c:a", "pcm_s16le", str(source),
    ], check=True)
    cuts = [Range(i + .103, i + .153) for i in range(24)]
    apply_cuts(source, cuts, out, validation="full")
    samples = audio_samples(out)
    window = 240  # 10 ms RMS windows, robust to AAC ringing at the join.
    rms = np.sqrt(np.mean(samples[:len(samples)//window*window].reshape(-1, window)**2, axis=1))
    onset = np.flatnonzero(rms > .05)[0] * .01
    expected = 28 - sum(c.duration for c in cuts)
    assert abs(onset - expected) < .025, (onset, expected)
    frames = subprocess.check_output([
        "ffmpeg", "-v", "error", "-i", str(out), "-map", "0:v:0",
        "-pix_fmt", "gray", "-f", "rawvideo", "-",
    ])
    luminance = np.frombuffer(frames, dtype=np.uint8).reshape(-1, 64 * 64).mean(axis=1)
    fps = float(Fraction(rate))
    source_frames = subprocess.check_output([
        "ffmpeg", "-v", "error", "-i", str(source), "-pix_fmt", "gray", "-f", "rawvideo", "-",
    ])
    original = np.frombuffer(source_frames, dtype=np.uint8).reshape(-1, 64 * 64).mean(axis=1)
    expected_visual = np.flatnonzero(original > 200)[0] / fps - sum(c.duration for c in cuts)
    visual_onset = np.flatnonzero(luminance > 200)[0] / fps
    assert abs(visual_onset - expected_visual) < 1 / fps


def test_full_render_keeps_later_cuts_mutes_speech_and_subtitles_aligned(tmp_path):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("Requires real FFmpeg")
    from cleancut.config import Config
    from cleancut.edl import EditDecision, EditDecisionList
    from cleancut.pipeline import PipelineOptions, render
    from cleancut.speech import SpeechClip

    source, out, word = tmp_path / "source.mkv", tmp_path / "out.mp4", tmp_path / "word.wav"
    subprocess.run([
        "ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
        ("color=c=black:s=64x64:r=24:d=30,"
         "drawbox=c=white:t=fill:enable='between(t,25.2,25.4)'"), "-f", "lavfi", "-i",
        "sine=frequency=440:sample_rate=24000:duration=30",
        "-c:v", "libx264", "-c:a", "pcm_s16le", str(source),
    ], check=True)
    subprocess.run([
        "ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
        "sine=frequency=1000:sample_rate=24000:duration=0.4", str(word),
    ], check=True)
    cuts = [Range(i + .103, i + .153) for i in range(24)] + [Range(25.1, 25.6)]
    edl = EditDecisionList(decisions=[
        *[EditDecision(c.start, c.end, "cut", "nudity") for c in cuts],
        EditDecision(26.5, 26.9, "mute", "profanity"),
        EditDecision(28, 28.4, "mute", "profanity"),
    ])
    subs = [Subtitle(1, 28, 28.4, "replacement")]
    shifted = shift_after_cuts(28, cuts)
    # Stub inference only: cut rendering, timing shifts, mixing, subtitles and
    # full media validation all run for real.
    with patch("cleancut.speech.prepare_replacements", return_value=[
        SpeechClip(shifted, shifted + .4, word),
    ]):
        render(edl, subs, PipelineOptions(video=source, output=out, soft_subs=True),
               Config(profanity_audio="replace", encoder="libx264", render_validation="full"))
    samples = audio_samples(out)

    def portion(start, duration=.1):
        return samples[round(start * 24000):round((start + duration) * 24000)]

    mute_at = shift_after_cuts(26.5, cuts)
    assert np.sqrt(np.mean(portion(mute_at + .15)**2)) < .003
    generated = portion(shifted + .15)
    dominant = np.argmax(abs(np.fft.rfft(generated))) * 24000 / len(generated)
    assert abs(dominant - 1000) < 20
    untouched = portion(shifted + .6)
    dominant = np.argmax(abs(np.fft.rfft(untouched))) * 24000 / len(untouched)
    assert abs(dominant - 440) < 20
    frames = subprocess.check_output([
        "ffmpeg", "-v", "error", "-i", str(out), "-pix_fmt", "gray", "-f", "rawvideo", "-",
    ])
    assert np.frombuffer(frames, dtype=np.uint8).max() < 40  # later scene cut removed white frames
    subtitle_text = subprocess.check_output([
        "ffmpeg", "-v", "error", "-i", str(out), "-map", "0:s:0", "-f", "srt", "-",
    ]).decode()
    from cleancut.subtitles import read_srt

    srt = tmp_path / "out.srt"
    srt.write_text(subtitle_text)
    assert abs(read_srt(srt)[0].start - shifted) < .002
