"""Unit tests for pipeline helpers that don't require external models."""
from __future__ import annotations

from unittest.mock import patch
from pathlib import Path

from cleancut.edl import EditDecisionList, EditDecision


def test_snap_edl_to_shots_empty_shots():
    """_snap_edl_to_shots returns unchanged EDL when no shots are available."""
    from cleancut.pipeline import _snap_edl_to_shots

    edl = EditDecisionList()
    edl.add(EditDecision(start=1.0, end=5.0, action="cut", category="sex"))
    result = _snap_edl_to_shots(edl, [])
    assert len(result) == len(edl)


def test_build_edl_skips_failed_visual_detector():
    """If the visual detector raises RuntimeError, build_edl should skip it and continue."""
    from cleancut.pipeline import PipelineOptions, build_edl
    from cleancut.config import Config

    config = Config.load_defaults()
    config.llm_enabled = False
    config.vlm_enabled = False
    config.audio_events_enabled = False
    config.density_enabled = False

    opts = PipelineOptions(
        video=Path("/nonexistent/video.mp4"),
        use_visual=True,
        use_whisper=False,
        use_scenes=False,
    )

    # Patch _get_subtitles_and_words to return empty (no Whisper call needed)
    with patch("cleancut.pipeline._get_subtitles_and_words", return_value=([], [])):
        # Patch scan_video in visual to raise RuntimeError (e.g. NudeNet not installed)
        with patch("cleancut.visual.scan_video", side_effect=RuntimeError("NudeNet not installed")):
            # Should not raise; build_edl catches RuntimeError from detectors
            edl, subs = build_edl(opts, config)
            assert isinstance(edl, EditDecisionList)


def test_build_edl_skips_failed_vlm_detector():
    """If the VLM detector raises RuntimeError, build_edl should skip it and continue."""
    from cleancut.pipeline import PipelineOptions, build_edl
    from cleancut.config import Config
    from cleancut.scenes import Shot

    config = Config.load_defaults()
    config.llm_enabled = False
    config.vlm_enabled = True
    config.audio_events_enabled = False
    config.density_enabled = False

    opts = PipelineOptions(
        video=Path("/nonexistent/video.mp4"),
        use_visual=False,
        use_whisper=False,
        use_scenes=False,
    )

    fake_shots = [Shot(start=0.0, end=5.0)]

    with patch("cleancut.pipeline._get_subtitles_and_words", return_value=([], [])):
        with patch("cleancut.pipeline._detect_scenes_if_enabled", return_value=fake_shots):
            with patch(
                "cleancut.classify_visual.scan_with_vlm",
                side_effect=RuntimeError("ollama not installed"),
            ):
                edl, subs = build_edl(opts, config)
                assert isinstance(edl, EditDecisionList)
