"""Tests for report.py — plan and results text rendering."""

from pathlib import Path

from cleancut.config import Config
from cleancut.edl import EditDecision, EditDecisionList
from cleancut.probe import Stream
from cleancut.report import build_plan, build_results_report, write_report


def _video_stream():
    return Stream(index=0, codec_name="h264", codec_type="video")


def _audio(idx, lang):
    return Stream(index=idx, codec_name="aac", codec_type="audio",
                  language=lang, channels=2)


def test_build_plan_includes_source_path(tmp_path):
    config = Config.load_defaults()
    plan = build_plan(
        video=Path("/movies/X.mp4"), streams=[_video_stream(), _audio(1, "eng")],
        config=config, audio_track=None, prefer_language="eng",
        use_visual=True, use_scenes=True, use_whisper=True, burn_subs=True,
    )
    assert "/movies/X.mp4" in plan
    assert "CLEANCUT PLAN" in plan


def test_build_plan_shows_explicit_subs(tmp_path):
    config = Config.load_defaults()
    plan = build_plan(
        video=Path("/x.mp4"), streams=[],
        config=config, audio_track=None, prefer_language="eng",
        use_visual=True, use_scenes=True, use_whisper=True, burn_subs=True,
        explicit_subs=Path("/custom.srt"),
    )
    assert "/custom.srt" in plan


def test_build_plan_marks_image_subs_skipped(tmp_path):
    config = Config.load_defaults()
    image_sub = Stream(index=2, codec_name="dvd_subtitle", codec_type="subtitle", language="eng")
    plan = build_plan(
        video=Path("/x.mp4"), streams=[_video_stream(), _audio(1, "eng"), image_sub],
        config=config, audio_track=None, prefer_language="eng",
        use_visual=True, use_scenes=True, use_whisper=True, burn_subs=True,
    )
    assert "image" in plan.lower()


def test_build_plan_disables_categories_visible():
    config = Config.load_defaults()
    config.enabled_categories.discard("violence")
    plan = build_plan(
        video=Path("/x.mp4"), streams=[_video_stream()],
        config=config, audio_track=None, prefer_language="eng",
        use_visual=True, use_scenes=True, use_whisper=True, burn_subs=True,
    )
    assert "violence" in plan
    # The violence row should say disabled
    violence_line = [line for line in plan.splitlines() if "violence" in line][0]
    assert "disabled" in violence_line


def test_build_results_report_summary_counts():
    edl = EditDecisionList(decisions=[
        EditDecision(start=10, end=15, action="cut", category="sex"),
        EditDecision(start=30, end=32, action="mute", category="profanity",
                     text_before="fuck off", text_after="freak off"),
    ])
    rep = build_results_report(Path("/x.mp4"), Path("/out.mp4"), edl,
                               original_duration=1000)
    assert "Total decisions: 2" in rep
    assert "Mutes:           1" in rep
    assert "Cuts:            1" in rep
    assert "Original length" in rep


def test_build_results_report_empty_edl():
    edl = EditDecisionList()
    rep = build_results_report(Path("/x.mp4"), None, edl)
    assert "No decisions" in rep


def test_build_results_report_shows_dialogue_softening():
    edl = EditDecisionList(decisions=[
        EditDecision(start=0, end=5, action="mute", category="profanity",
                     text_before="what the hell", text_after="what the heck"),
    ])
    rep = build_results_report(Path("/x.mp4"), None, edl)
    assert "what the hell" in rep
    assert "what the heck" in rep


def test_write_report_roundtrip(tmp_path):
    out = tmp_path / "r.txt"
    write_report("hello\nworld\n", out)
    assert out.read_text() == "hello\nworld\n"


def test_build_results_report_groups_by_category():
    edl = EditDecisionList(decisions=[
        EditDecision(start=0, end=2, action="cut", category="drugs"),
        EditDecision(start=10, end=12, action="cut", category="drugs"),
        EditDecision(start=20, end=22, action="cut", category="sex"),
    ])
    rep = build_results_report(Path("/x.mp4"), None, edl)
    # By-category section should show "drugs  2  " and "sex  1  "
    assert "drugs" in rep
    assert "sex" in rep
