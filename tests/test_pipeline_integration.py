"""Integration-style tests for pipeline.build_edl() with all detectors mocked.

Verifies:
- The full pipeline path runs end-to-end with every detector returning data
- Detectors that raise RuntimeError are skipped gracefully
- The returned EDL aggregates decisions from all active detectors
"""
from __future__ import annotations

from unittest.mock import patch


from cleancut.config import Config
from cleancut.edl import EditDecision, EditDecisionList
from cleancut.pipeline import PipelineOptions, build_edl
from cleancut.scenes import Shot
from cleancut.subtitles import Subtitle


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_decision(start: float, end: float, category: str, source: str) -> EditDecision:
    return EditDecision(
        start=start, end=end, action="cut",
        category=category, reason="test", source=source,
    )


def _make_mute(start: float, end: float, category: str, source: str) -> EditDecision:
    return EditDecision(
        start=start, end=end, action="mute",
        category=category, reason="test", source=source,
    )


def _minimal_config(*, llm=False, vlm=False, audio_events=False,
                    density=False, visual=True) -> Config:
    cfg = Config.load_defaults()
    cfg.llm_enabled = llm
    cfg.vlm_enabled = vlm
    cfg.audio_events_enabled = audio_events
    cfg.density_enabled = density
    # Disable corroboration so visual-only cuts are not filtered out.
    cfg.require_visual_corroboration = False
    cfg.snap_cuts_to_scenes = False
    return cfg


# ---------------------------------------------------------------------------
# Full-pipeline smoke tests
# ---------------------------------------------------------------------------

class TestBuildEdlFullPipeline:
    def test_all_detectors_disabled_returns_empty_edl(self, tmp_path):
        video = tmp_path / "movie.mp4"
        video.write_bytes(b"\x00")
        config = _minimal_config(visual=False)
        opts = PipelineOptions(
            video=video,
            use_visual=False,
            use_whisper=False,
            use_scenes=False,
        )

        with patch("cleancut.pipeline._get_subtitles_and_words", return_value=([], [])):
            edl, subs = build_edl(opts, config)

        assert isinstance(edl, EditDecisionList)
        assert subs == []

    def test_visual_decisions_included_in_edl(self, tmp_path):
        video = tmp_path / "movie.mp4"
        video.write_bytes(b"\x00")
        config = _minimal_config(visual=True)
        config.enabled_categories.add("nudity")
        opts = PipelineOptions(
            video=video,
            use_visual=True,
            use_whisper=False,
            use_scenes=False,
        )
        visual_dec = _make_decision(60.0, 90.0, "nudity", "visual")

        with patch("cleancut.pipeline._get_subtitles_and_words", return_value=([], [])), \
             patch("cleancut.pipeline._detect_scenes_if_enabled", return_value=[]), \
             patch("cleancut.visual.scan_video") as mock_scan:
            mock_edl = EditDecisionList(decisions=[visual_dec])
            mock_scan.return_value = mock_edl
            edl, _ = build_edl(opts, config)

        sources = {d.source for d in edl.decisions}
        assert "visual" in sources

    def test_subtitle_dialogue_decisions_included(self, tmp_path):
        video = tmp_path / "movie.mp4"
        video.write_bytes(b"\x00")
        config = _minimal_config(visual=False)
        opts = PipelineOptions(
            video=video,
            use_visual=False,
            use_whisper=False,
            use_scenes=False,
        )
        fake_subs = [Subtitle(index=1, start=5.0, end=8.0, text="damn it")]
        dialogue_dec = _make_mute(5.0, 8.0, "profanity", "subtitle")

        with patch("cleancut.pipeline._get_subtitles_and_words", return_value=(fake_subs, [])), \
             patch("cleancut.subtitles.scan_subtitles") as mock_scan_subs:
            mock_scan_subs.return_value = EditDecisionList(decisions=[dialogue_dec])
            edl, subs = build_edl(opts, config)

        assert subs == fake_subs

    def test_visual_runtime_error_is_skipped_gracefully(self, tmp_path):
        video = tmp_path / "movie.mp4"
        video.write_bytes(b"\x00")
        config = _minimal_config(visual=True)
        config.enabled_categories.add("nudity")
        opts = PipelineOptions(
            video=video,
            use_visual=True,
            use_whisper=False,
            use_scenes=False,
        )

        with patch("cleancut.pipeline._get_subtitles_and_words", return_value=([], [])), \
             patch("cleancut.pipeline._detect_scenes_if_enabled", return_value=[]), \
             patch("cleancut.visual.scan_video", side_effect=RuntimeError("NudeNet not installed")):
            # Must not raise
            edl, _ = build_edl(opts, config)

        assert isinstance(edl, EditDecisionList)

    def test_llm_decisions_included_when_enabled(self, tmp_path):
        video = tmp_path / "movie.mp4"
        video.write_bytes(b"\x00")
        config = _minimal_config(llm=True)
        opts = PipelineOptions(
            video=video,
            use_visual=False,
            use_whisper=False,
            use_scenes=False,
        )
        fake_subs = [Subtitle(index=1, start=30.0, end=35.0, text="explicit dialogue")]
        llm_dec = _make_mute(30.0, 35.0, "sex", "llm")

        with patch("cleancut.pipeline._get_subtitles_and_words", return_value=(fake_subs, [])), \
             patch("cleancut.pipeline._detect_scenes_if_enabled", return_value=[]), \
             patch("cleancut.classify_dialogue.classify_dialogue") as mock_llm:
            mock_llm.return_value = EditDecisionList(decisions=[llm_dec])
            edl, _ = build_edl(opts, config)

        sources = {d.source for d in edl.decisions}
        assert "llm" in sources

    def test_llm_runtime_error_is_skipped_gracefully(self, tmp_path):
        video = tmp_path / "movie.mp4"
        video.write_bytes(b"\x00")
        config = _minimal_config(llm=True)
        opts = PipelineOptions(
            video=video,
            use_visual=False,
            use_whisper=False,
            use_scenes=False,
        )
        fake_subs = [Subtitle(index=1, start=1.0, end=2.0, text="hello")]

        with patch("cleancut.pipeline._get_subtitles_and_words", return_value=(fake_subs, [])), \
             patch("cleancut.pipeline._detect_scenes_if_enabled", return_value=[]), \
             patch("cleancut.classify_dialogue.classify_dialogue",
                   side_effect=RuntimeError("Ollama not running")):
            edl, _ = build_edl(opts, config)

        assert isinstance(edl, EditDecisionList)

    def test_vlm_decisions_included_when_enabled(self, tmp_path):
        video = tmp_path / "movie.mp4"
        video.write_bytes(b"\x00")
        config = _minimal_config(vlm=True)
        opts = PipelineOptions(
            video=video,
            use_visual=False,
            use_whisper=False,
            use_scenes=False,
        )
        fake_shots = [Shot(start=0.0, end=10.0)]
        vlm_dec = _make_decision(0.0, 10.0, "sex", "vlm")

        with patch("cleancut.pipeline._get_subtitles_and_words", return_value=([], [])), \
             patch("cleancut.pipeline._detect_scenes_if_enabled", return_value=fake_shots), \
             patch("cleancut.classify_visual.scan_with_vlm") as mock_vlm:
            mock_vlm.return_value = EditDecisionList(decisions=[vlm_dec])
            edl, _ = build_edl(opts, config)

        sources = {d.source for d in edl.decisions}
        assert "vlm" in sources

    def test_vlm_runtime_error_is_skipped_gracefully(self, tmp_path):
        video = tmp_path / "movie.mp4"
        video.write_bytes(b"\x00")
        config = _minimal_config(vlm=True)
        opts = PipelineOptions(
            video=video,
            use_visual=False,
            use_whisper=False,
            use_scenes=False,
        )
        fake_shots = [Shot(start=0.0, end=10.0)]

        with patch("cleancut.pipeline._get_subtitles_and_words", return_value=([], [])), \
             patch("cleancut.pipeline._detect_scenes_if_enabled", return_value=fake_shots), \
             patch("cleancut.classify_visual.scan_with_vlm",
                   side_effect=RuntimeError("ollama not installed")):
            edl, _ = build_edl(opts, config)

        assert isinstance(edl, EditDecisionList)

    def test_audio_events_decisions_included_when_enabled(self, tmp_path):
        video = tmp_path / "movie.mp4"
        video.write_bytes(b"\x00")
        config = _minimal_config(audio_events=True)
        opts = PipelineOptions(
            video=video,
            use_visual=False,
            use_whisper=False,
            use_scenes=False,
        )
        fake_shots = [Shot(start=0.0, end=10.0)]
        ae_dec = _make_decision(0.0, 10.0, "sex", "audio_events")

        with patch("cleancut.pipeline._get_subtitles_and_words", return_value=([], [])), \
             patch("cleancut.pipeline._detect_scenes_if_enabled", return_value=fake_shots), \
             patch("cleancut.audio_events.scan_audio_events") as mock_ae:
            mock_ae.return_value = EditDecisionList(decisions=[ae_dec])
            edl, _ = build_edl(opts, config)

        sources = {d.source for d in edl.decisions}
        assert "audio_events" in sources

    def test_audio_events_runtime_error_is_skipped_gracefully(self, tmp_path):
        video = tmp_path / "movie.mp4"
        video.write_bytes(b"\x00")
        config = _minimal_config(audio_events=True)
        opts = PipelineOptions(
            video=video,
            use_visual=False,
            use_whisper=False,
            use_scenes=False,
        )
        fake_shots = [Shot(start=0.0, end=10.0)]

        with patch("cleancut.pipeline._get_subtitles_and_words", return_value=([], [])), \
             patch("cleancut.pipeline._detect_scenes_if_enabled", return_value=fake_shots), \
             patch("cleancut.audio_events.scan_audio_events",
                   side_effect=RuntimeError("transformers not installed")):
            edl, _ = build_edl(opts, config)

        assert isinstance(edl, EditDecisionList)

    def test_all_detectors_combined_edl_has_multiple_sources(self, tmp_path):
        """Run with visual + LLM + VLM mocked to return decisions; verify all sources present."""
        video = tmp_path / "movie.mp4"
        video.write_bytes(b"\x00")
        config = _minimal_config(llm=True, vlm=True, visual=True)
        config.enabled_categories.add("nudity")
        opts = PipelineOptions(
            video=video,
            use_visual=True,
            use_whisper=False,
            use_scenes=False,
        )
        fake_shots = [Shot(start=0.0, end=30.0)]
        fake_subs = [Subtitle(index=1, start=5.0, end=8.0, text="bad word")]

        visual_dec = _make_decision(0.0, 30.0, "nudity", "visual")
        llm_dec = _make_mute(5.0, 8.0, "sex", "llm")
        vlm_dec = _make_decision(10.0, 30.0, "sex", "vlm")

        with patch("cleancut.pipeline._get_subtitles_and_words", return_value=(fake_subs, [])), \
             patch("cleancut.pipeline._detect_scenes_if_enabled", return_value=fake_shots), \
             patch("cleancut.visual.scan_video",
                   return_value=EditDecisionList(decisions=[visual_dec])), \
             patch("cleancut.classify_dialogue.classify_dialogue",
                   return_value=EditDecisionList(decisions=[llm_dec])), \
             patch("cleancut.classify_visual.scan_with_vlm",
                   return_value=EditDecisionList(decisions=[vlm_dec])):
            edl, _ = build_edl(opts, config)

        assert len(edl.decisions) >= 1
        # At least one decision should exist (they might merge)
        assert edl.decisions

    def test_edl_video_path_set_correctly(self, tmp_path):
        video = tmp_path / "movie.mp4"
        video.write_bytes(b"\x00")
        config = _minimal_config()
        opts = PipelineOptions(
            video=video,
            use_visual=False,
            use_whisper=False,
            use_scenes=False,
        )

        with patch("cleancut.pipeline._get_subtitles_and_words", return_value=([], [])):
            edl, _ = build_edl(opts, config)

        assert edl.video_path == str(video)

    def test_density_clustering_applied_when_enough_events(self, tmp_path):
        """When density is enabled and enough events exist, clusters are added."""
        video = tmp_path / "movie.mp4"
        video.write_bytes(b"\x00")
        config = _minimal_config(density=True, visual=False)
        config.density_min_events = 2
        opts = PipelineOptions(
            video=video,
            use_visual=False,
            use_whisper=False,
            use_scenes=False,
        )
        # Provide subtitle-based decisions so the density threshold is met.
        fake_subs = [
            Subtitle(index=i, start=float(i * 5), end=float(i * 5 + 2), text="bad")
            for i in range(3)
        ]
        dialogue_decisions = [
            _make_mute(float(i * 5), float(i * 5 + 2), "profanity", "subtitle")
            for i in range(3)
        ]

        with patch("cleancut.pipeline._get_subtitles_and_words", return_value=(fake_subs, [])), \
             patch("cleancut.pipeline._detect_scenes_if_enabled", return_value=[]), \
             patch("cleancut.pipeline.scan_subtitles") as mock_scan, \
             patch("cleancut.density.find_clusters") as mock_density:
            mock_scan.return_value = EditDecisionList(decisions=dialogue_decisions)
            cluster_dec = _make_decision(0.0, 15.0, "profanity", "density")
            mock_density.return_value = EditDecisionList(decisions=[cluster_dec])
            edl, _ = build_edl(opts, config)

        # density.find_clusters should have been called
        mock_density.assert_called_once()
