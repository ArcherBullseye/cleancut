"""Regression tests for CLI/UX and matching bugs found in the July 2026 review."""
from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import patch

import pytest

from cleancut.config import Config
from cleancut.edl import EditDecision, EditDecisionList
from cleancut.subtitles import Subtitle, scan_subtitles, soften_text


def _decision(start, end, action="mute", category="profanity", source="subtitle",
              accepted=True, reason="matched: x"):
    return EditDecision(start=start, end=end, action=action, category=category,
                        source=source, accepted=accepted, reason=reason)


class TestSoftenText:
    def test_replacements_do_not_chain(self):
        """With shit→crap and crap→crud, "shit" must become "crap", not "crud"
        (the old sequential pass re-substituted its own output)."""
        replacements = {"shit": "crap", "crap": "crud"}
        assert soften_text("Holy shit, that's crap.", replacements) == \
            "Holy crap, that's crud."

    def test_case_preserved(self):
        assert soften_text("Shit happens", {"shit": "crap"}) == "Crap happens"
        assert soften_text("SHIT!", {"shit": "crap"}) == "CRAP!"

    def test_longer_phrase_wins(self):
        replacements = {"god damn": "gosh darn", "damn": "darn"}
        assert soften_text("god damn it", replacements) == "gosh darn it"


class TestWordlistProperNouns:
    def test_dickens_and_dickinson_not_flagged(self):
        config = Config.load_defaults()
        subs = [Subtitle(index=1, start=0, end=2,
                         text="Charles Dickens admired Emily Dickinson.")]
        edl = scan_subtitles(subs, config)
        assert len(edl) == 0

    def test_actual_profanity_still_flagged(self):
        config = Config.load_defaults()
        subs = [Subtitle(index=1, start=0, end=2, text="What a dickhead.")]
        edl = scan_subtitles(subs, config)
        assert len(edl) == 1


class TestEdlSemantics:
    def test_summary_counts_accepted_only(self):
        """The summary printed after review must agree with what renders."""
        edl = EditDecisionList(decisions=[
            _decision(0, 1, action="cut", category="nudity"),
            _decision(5, 6, action="cut", category="nudity", accepted=False),
        ])
        assert edl.summary() == {"cut:nudity": 1}

    def test_merge_overlapping_does_not_mutate_input(self):
        d1 = _decision(0, 10, action="cut", category="sex")
        d2 = _decision(5, 20, action="cut", category="sex")
        edl = EditDecisionList(decisions=[d1, d2])
        merged = edl.merge_overlapping()
        assert d1.end == 10  # input object untouched
        assert merged.decisions[0].end == 20


class TestDensity:
    def test_cluster_end_uses_max_end_not_last_start(self):
        """Events are sorted by start; an early long event can end after the
        last-starting one and must still bound the cluster."""
        from cleancut.density import DensityParams, find_clusters
        edl = EditDecisionList(decisions=[
            _decision(0, 50),    # long event — latest end
            _decision(1, 2),
            _decision(3, 4),
        ])
        out = find_clusters(edl, DensityParams(window_seconds=60, min_events=3,
                                               min_cluster_span=8))
        assert len(out) == 1
        assert out.decisions[0].end >= 50

    def test_density_clusters_llm_decisions(self):
        """Density lists llm-dialogue as a cluster source, so it must run AFTER
        the LLM detector, not before."""
        from cleancut.pipeline import PipelineOptions, build_edl

        subs = [Subtitle(index=i, start=i * 5.0, end=i * 5.0 + 2, text="clean line")
                for i in range(1, 13)]
        llm_edl = EditDecisionList(decisions=[
            _decision(10, 15, action="cut", category="drugs", source="llm-dialogue"),
            _decision(30, 35, action="cut", category="drugs", source="llm-dialogue"),
            _decision(50, 58, action="cut", category="drugs", source="llm-dialogue"),
        ])
        config = Config.load_defaults()
        config.llm_enabled = True
        config.density_enabled = True
        config.vlm_enabled = False
        config.audio_events_enabled = False

        opts = PipelineOptions(video=Path("/fake.mp4"), use_visual=False,
                               use_whisper=False, use_scenes=False)
        with patch("cleancut.pipeline._get_subtitles_and_words", return_value=(subs, [])), \
             patch("cleancut.classify_dialogue.classify_dialogue", return_value=llm_edl):
            edl, _ = build_edl(opts, config)

        assert any("density" in d.source for d in edl.decisions)


class TestPipelineSubsValidation:
    def test_missing_explicit_subs_is_an_error(self, tmp_path):
        """A typo'd --subs path must not silently fall through to an
        hours-long Whisper run."""
        from cleancut.pipeline import PipelineOptions, _get_subtitles_and_words

        opts = PipelineOptions(video=tmp_path / "movie.mp4",
                               subs=tmp_path / "typo.srt")
        with pytest.raises(ValueError, match="typo.srt"):
            _get_subtitles_and_words(opts, Config.load_defaults())


class TestActionValidation:
    def _args(self, **extra):
        defaults = dict(
            preset=None, wordlists=None, replacements=None,
            enable_category=None, disable_category=None, action=None,
            no_word_timestamps=False, no_snap_to_scenes=False,
            vlm_cut_intimate=False, allow_solo_visual=False,
        )
        defaults.update(extra)
        return argparse.Namespace(**defaults)

    def test_action_with_unknown_category_rejected(self):
        from cleancut.cli import _apply_common
        with pytest.raises(SystemExit):
            _apply_common(self._args(action=["porfanity=mute"]), Config.load_defaults())

    def test_action_with_valid_category_accepted(self):
        from cleancut.cli import _apply_common
        config = Config.load_defaults()
        _apply_common(self._args(action=["profanity=cut"]), config)
        assert config.actions["profanity"] == "cut"


class TestReviewCommand:
    def _edl_with_cuts(self, tmp_path, n=1, video_path=None):
        video = tmp_path / "movie.mp4"
        video.write_bytes(b"\x00")
        edl_path = tmp_path / "movie.edl.json"
        edl = EditDecisionList(
            decisions=[
                EditDecision(start=10.0 + 20 * i, end=20.0 + 20 * i, action="cut",
                             category="nudity", source="visual", reason="test")
                for i in range(n)
            ],
            video_path=str(video) if video_path is None else video_path,
        )
        edl.to_json(edl_path)
        return edl_path, video

    def _args(self, edl_path, video=None, **extra):
        defaults = dict(edl=str(edl_path), video=str(video) if video else None,
                        subs=None, frames_dir=None, include_violence=False)
        defaults.update(extra)
        return argparse.Namespace(**defaults)

    def _run(self, args, inputs):
        with patch("subprocess.run"), \
             patch("builtins.input", side_effect=inputs), \
             patch("cleancut.editor.probe_duration", return_value=None), \
             patch("cleancut.cli.write_report"), \
             patch("cleancut.cli.build_results_report", return_value=""):
            from cleancut.cli import cmd_review
            return cmd_review(args)

    def test_trim_rejects_end_before_start(self, tmp_path):
        edl_path, video = self._edl_with_cuts(tmp_path)
        result = self._run(self._args(edl_path, video), ["t 0:20 0:10", "q"])
        assert result == 0
        loaded = EditDecisionList.from_json(edl_path)
        assert (loaded.decisions[0].start, loaded.decisions[0].end) == (10.0, 20.0)

    def test_trim_is_atomic_on_bad_end_timestamp(self, tmp_path):
        """A bad END must not leave a half-applied trim (new start, old end)."""
        edl_path, video = self._edl_with_cuts(tmp_path)
        result = self._run(self._args(edl_path, video), ["t 0:05 zz:zz", "q"])
        assert result == 0
        loaded = EditDecisionList.from_json(edl_path)
        assert loaded.decisions[0].start == 10.0

    def test_eof_saves_progress(self, tmp_path):
        """Ctrl-D / closed stdin must save the review work done so far."""
        edl_path, video = self._edl_with_cuts(tmp_path, n=2)
        result = self._run(self._args(edl_path, video), ["n", EOFError()])
        assert result == 0
        loaded = EditDecisionList.from_json(edl_path)
        assert loaded.decisions[0].accepted is False

    def test_empty_video_path_in_edl_is_an_error(self, tmp_path):
        """video_path="" must not resolve to Path('.') and 'succeed'."""
        edl_path, _ = self._edl_with_cuts(tmp_path, video_path="")
        result = self._run(self._args(edl_path, video=None), ["q"])
        assert result == 1


class TestSidecarLanguagePreference:
    def test_prefer_language_spa_picks_spanish_srt(self, tmp_path):
        from cleancut.probe import find_sidecar_subtitle
        video = tmp_path / "Movie.mp4"
        video.write_bytes(b"\x00")
        (tmp_path / "Movie.en.srt").write_text("1\n00:00:01,000 --> 00:00:02,000\nhi\n")
        (tmp_path / "Movie.spa.srt").write_text("1\n00:00:01,000 --> 00:00:02,000\nhola\n")

        assert find_sidecar_subtitle(video, prefer_language="spa") == tmp_path / "Movie.spa.srt"
        assert find_sidecar_subtitle(video, prefer_language="eng") == tmp_path / "Movie.en.srt"
