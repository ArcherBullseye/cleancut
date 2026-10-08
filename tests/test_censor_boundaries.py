"""Regressions for leaked word endings and repeated weak NudeNet false positives."""
from __future__ import annotations

import shutil
import subprocess
import wave
from unittest.mock import patch

import numpy as np
import pytest

from cleancut.config import Config
from cleancut.edl import EditDecision, EditDecisionList, snap_edl_to_shots
from cleancut.scenes import Shot
from cleancut.subtitles import scan_words
from cleancut.transcribe import Word
from cleancut.visual import _confirmed_edl, _iter_sampled_frames, _observations_by_frame


@pytest.mark.parametrize("action", ["mute", "replace", "cut"])
@pytest.mark.parametrize("following", [1.8, 1.28, 1.2, 1.1, None])
def test_word_ending_guard_cannot_be_cancelled_by_uncertain_next_word(action, following):
    cfg = Config.load_defaults()
    cfg.actions["profanity"] = action
    words = [Word(1, 1.2, "fuck")]
    if following is not None:
        words.append(Word(following, following+.3, "that"))
    edl = scan_words(words, cfg).pad(.15)
    d = edl.decisions[0]
    assert d.start == pytest.approx(.96)
    assert d.end == pytest.approx(1.4)
    assert d.word_edits[0]["end"] == 1.2  # synthesized word length isn't stretched by the guard
    assert d.word_edits[0]["next_word_start"] == following


def test_buffer_is_configurable_and_no_buffer_is_respected():
    raw = scan_words([Word(1, 1.2, "fuck")], Config.load_defaults())
    assert raw.pad(.15, word_end_padding_ms=350).decisions[0].end == pytest.approx(1.55)
    assert raw.pad(.15, word_end_padding_ms=0).decisions[0].end == 1.2
    assert raw.decisions[0].end == 1.2


@pytest.mark.parametrize("start,count,score,anatomy", [
    (30*60+26, 19, .54, "MALE_GENITALIA_EXPOSED"),
    (22*60+25, 7, .65, "FEMALE_BREAST_EXPOSED"),
    (14*60+52, 4, .52, "MALE_GENITALIA_EXPOSED"),
])
@pytest.mark.parametrize("preset", ["fast", "balanced", "thorough"])
def test_reported_leverage_scores_cannot_create_automatic_cuts(start, count, score, anatomy, preset):
    cfg = Config.load_defaults()
    cfg.apply_preset(preset)
    observations = {start + i/6: [{"class": anatomy, "score": score}] for i in range(count)}
    edl = _confirmed_edl(observations, 2500, cfg, "cut", object())
    assert edl.by_action("cut") == []
    assert all(not d.accepted for d in edl)
    if score >= cfg.visual_threshold:
        assert len(edl) == 1 and "not auto-cut" in edl.decisions[0].reason


def test_repeated_same_class_confident_nudity_and_brief_strong_nudity_survive():
    cfg = Config()
    observations = {5+i/6: [{"class": "FEMALE_BREAST_EXPOSED", "score": .78}] for i in range(3)}
    observations[9] = [{"class": "MALE_GENITALIA_EXPOSED", "score": .94}]
    edl = _confirmed_edl(observations, 20, cfg, "cut", object())
    assert len(edl.by_action("cut")) == 2
    assert all(d.accepted for d in edl)


def test_different_anatomy_classes_and_duplicate_boxes_do_not_add_confirmations():
    observations = {
        5: [{"class": "FEMALE_BREAST_EXPOSED", "score": .78}]*5,
        5.2: [{"class": "MALE_GENITALIA_EXPOSED", "score": .78}],
        5.4: [{"class": "BUTTOCKS_EXPOSED", "score": .78}],
    }
    assert not _confirmed_edl(observations, 20, Config(), "cut", object()).by_action("cut")


def test_weak_hits_do_not_extend_a_strong_detection_across_the_entire_cluster():
    def hit(score):
        return [{"class": "FEMALE_BREAST_EXPOSED", "score": score}]
    observations = {i/2: hit(.5) for i in range(20)}
    observations[5] = hit(.95)
    edl = _confirmed_edl(observations, 20, Config(), "cut", object())
    assert [(d.start, d.end) for d in edl.by_action("cut")] == [(4.5, 5.5)]


def test_same_decoded_frame_cannot_be_counted_twice_by_coarse_and_dense_passes():
    hit = [{"class": "FEMALE_BREAST_EXPOSED", "score": .8}]
    observations = _observations_by_frame([(5, hit), (5.001, hit), (5.01, hit)], 24)
    assert len(observations) == 1
    assert not _confirmed_edl(observations, 20, Config(), "cut", object()).by_action("cut")

    class Capture:
        def __init__(self):
            self.reads = 0

        def read(self):
            self.reads += 1
            return True, self.reads

        def grab(self):
            return True

    capture = Capture()
    assert list(_iter_sampled_frames(capture, 24, [(0, 0), (.001, .001), (.04, .04)])) == [(0, 1), (.04, 2)]


@pytest.mark.parametrize("first_rejected", [False, True])
def test_review_only_nudity_never_changes_accepted_edits(first_rejected):
    mute = EditDecision(5, 5.4, "mute", "profanity", source="whisper-word")
    weak = EditDecision(4.8 if first_rejected else 5.2, 10, "cut", "nudity",
                        source="visual", accepted=False)
    second_mute = EditDecision(5.3, 5.5, "mute", "profanity", source="whisper-word")
    edl = EditDecisionList(decisions=[mute, weak, second_mute]).merge_overlapping(.5)
    assert not edl.by_action("cut")
    assert [(d.start, d.end) for d in edl.by_action("mute")] == [(5, 5.5)]
    assert len(edl) == 2


@pytest.mark.parametrize("source", ["visual", "visual-temporal", "whisper-word"])
def test_precise_cuts_are_not_expanded_to_entire_shots(source):
    d = EditDecision(5, 5.8, "cut", "nudity" if source != "whisper-word" else "profanity", source=source)
    edl = snap_edl_to_shots(EditDecisionList(decisions=[d]), [Shot(0, 10)])
    assert (edl.decisions[0].start, edl.decisions[0].end) == (5, 5.8)
    assert "snapped" not in edl.decisions[0].reason


@pytest.mark.parametrize("action", ["mute", "replace"])
@pytest.mark.parametrize("next_word_timestamp", [1.6, 1.2])
def test_real_render_silences_underestimated_word_tail_after_cut(tmp_path, action, next_word_timestamp):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("Requires FFmpeg")
    from cleancut.pipeline import PipelineOptions, render

    rate = 48000
    t = np.arange(rate*3)/rate
    # Whisper underestimates this word's end by 160 ms. The next word begins at 1.6 s.
    samples = (.2*np.sin(2*np.pi*440*t)*((t >= 1) & (t < 1.36))
               + .2*np.sin(2*np.pi*880*t)*((t >= 1.6) & (t < 2.4)))
    audio = tmp_path / "dialogue.wav"
    with wave.open(str(audio), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes((samples*32767).astype("<i2").tobytes())
    video = tmp_path / "source.mkv"
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-v", "error", "-f", "lavfi", "-i",
                    "color=c=black:s=64x64:r=24:d=3", "-i", str(audio), "-c:v", "libx264",
                    "-c:a", "pcm_s16le", str(video)], check=True)
    cfg = Config.load_defaults()
    cfg.encoder, cfg.render_validation = "libx264", "full"
    cfg.actions["profanity"] = action
    # Whisper can assign the remaining phoneme to the next word. Its onset
    # must not cancel the ending guard even though that timestamp is contiguous.
    edl = scan_words([Word(1, 1.2, "fuck"), Word(next_word_timestamp, 2.4, "that")], cfg).pad(.15)
    edl.add(EditDecision(.2, .4, "cut", "nudity"))
    out = tmp_path / "clean.mp4"
    with patch("cleancut.speech.check_service", side_effect=OSError("offline")):
        render(edl, [], PipelineOptions(video=video, output=out, burn_subs=False), cfg)
    decoded = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(out), "-vn",
                                       "-ac", "1", "-ar", str(rate), "-f", "s16le", "-"])
    data = np.frombuffer(decoded, dtype="<i2").astype(float)

    def rms(start, end):
        return np.sqrt(np.mean(data[round(start*rate):round(end*rate)]**2))

    assert rms(1.08, 1.15) < 20  # source 1.28–1.35: the previously audible tail
    assert rms(1.45, 1.65) > 3000  # the following word survives
