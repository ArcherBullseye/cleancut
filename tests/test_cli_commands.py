"""Tests for CLI subcommands: scan, clean, inspect, add-cut, review.

All external I/O (build_edl, run_full, probe_streams, etc.) is mocked so
no real video file or external tool is required.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import patch


from cleancut.edl import EditDecision, EditDecisionList
from cleancut.cli import cmd_scan, cmd_clean, cmd_inspect, cmd_add_cut, cmd_review


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_edl(n_decisions: int = 1, video_path: str = "/fake/video.mp4") -> EditDecisionList:
    decisions = [
        EditDecision(
            start=float(i * 10),
            end=float(i * 10 + 5),
            action="cut",
            category="nudity",
            source="visual",
            reason="test",
        )
        for i in range(n_decisions)
    ]
    return EditDecisionList(decisions=decisions, video_path=video_path)


def _args_via_parser(subcommand: str, tmp_path: Path, **extra) -> argparse.Namespace:
    """Build a Namespace through the REAL parser so defaults can never drift
    from the CLI, then apply per-test attribute overrides."""
    from cleancut.cli import build_parser

    video = tmp_path / "movie.mp4"
    video.write_bytes(b"\x00")
    ns = build_parser().parse_args(
        [subcommand, str(video), "--no-whisper", "--no-scenes"]
    )
    ns.preset = None  # tests exercise raw Config defaults unless they opt in
    for k, v in extra.items():
        setattr(ns, k, v)
    return ns


def _scan_args(tmp_path: Path, **extra) -> argparse.Namespace:
    return _args_via_parser("scan", tmp_path, **extra)


def _clean_args(tmp_path: Path, **extra) -> argparse.Namespace:
    return _args_via_parser("clean", tmp_path, **extra)


def _inspect_args(tmp_path: Path, **extra) -> argparse.Namespace:
    return _args_via_parser("inspect", tmp_path, **extra)


# ---------------------------------------------------------------------------
# cmd_scan
# ---------------------------------------------------------------------------

class TestCmdScan:
    def test_calls_build_edl_and_saves_edl(self, tmp_path):
        args = _scan_args(tmp_path)
        video_path = Path(args.video)
        fake_edl = _make_edl(video_path=str(video_path))

        with patch("cleancut.cli.build_edl", return_value=(fake_edl, [])) as mock_build, \
             patch("cleancut.editor.probe_duration", return_value=120.0), \
             patch("cleancut.cli.write_report"), \
             patch("cleancut.cli.build_results_report", return_value="report text"):
            result = cmd_scan(args)

        assert result == 0
        mock_build.assert_called_once()
        # Verify the EDL was written next to the video.
        expected_edl = video_path.with_suffix(".edl.json")
        assert expected_edl.exists()

    def test_edl_saved_to_explicit_output_path(self, tmp_path):
        out = tmp_path / "my_output.edl.json"
        args = _scan_args(tmp_path, output=str(out))
        fake_edl = _make_edl(video_path=args.video)

        with patch("cleancut.cli.build_edl", return_value=(fake_edl, [])), \
             patch("cleancut.editor.probe_duration", return_value=120.0), \
             patch("cleancut.cli.write_report"), \
             patch("cleancut.cli.build_results_report", return_value="report text"):
            result = cmd_scan(args)

        assert result == 0
        assert out.exists()

    def test_build_edl_receives_correct_video_path(self, tmp_path):
        args = _scan_args(tmp_path)
        video_path = Path(args.video)
        fake_edl = _make_edl(video_path=str(video_path))
        captured = {}

        def fake_build(opts, config):
            captured["opts"] = opts
            return fake_edl, []

        with patch("cleancut.cli.build_edl", side_effect=fake_build), \
             patch("cleancut.editor.probe_duration", return_value=None), \
             patch("cleancut.cli.write_report"), \
             patch("cleancut.cli.build_results_report", return_value=""):
            cmd_scan(args)

        assert captured["opts"].video == video_path

    def test_no_visual_flag_propagated(self, tmp_path):
        args = _scan_args(tmp_path, no_visual=True)
        fake_edl = _make_edl(video_path=args.video)
        captured = {}

        def fake_build(opts, config):
            captured["opts"] = opts
            return fake_edl, []

        with patch("cleancut.cli.build_edl", side_effect=fake_build), \
             patch("cleancut.editor.probe_duration", return_value=None), \
             patch("cleancut.cli.write_report"), \
             patch("cleancut.cli.build_results_report", return_value=""):
            cmd_scan(args)

        assert captured["opts"].use_visual is False


# ---------------------------------------------------------------------------
# cmd_clean
# ---------------------------------------------------------------------------

class TestCmdClean:
    def test_calls_run_full_and_returns_zero(self, tmp_path):
        args = _clean_args(tmp_path)
        video_path = Path(args.video)
        expected_out = video_path.with_name(f"{video_path.stem}.clean{video_path.suffix}")
        expected_out.write_bytes(b"\x00")  # fake output file

        with patch("cleancut.cli.run_full", return_value=(expected_out, _make_edl())) as mock_run, \
             patch("cleancut.cli.build_edl", return_value=(_make_edl(), [])) as mock_build, \
             patch("cleancut.editor.probe_duration", return_value=60.0), \
             patch("cleancut.cli.write_report"), \
             patch("cleancut.cli.build_results_report", return_value="report"):
            result = cmd_clean(args)

        assert result == 0
        mock_run.assert_called_once()
        # The report must reuse the EDL from run_full — a second build_edl call
        # would re-run the whole detection stack (Whisper/LLM/VLM) for nothing.
        mock_build.assert_not_called()

    def test_explicit_output_path_is_used(self, tmp_path):
        out = tmp_path / "clean_output.mp4"
        out.write_bytes(b"\x00")
        args = _clean_args(tmp_path, output=str(out))
        captured = {}

        def fake_run(opts, config):
            captured["opts"] = opts
            return out, _make_edl()

        with patch("cleancut.cli.run_full", side_effect=fake_run), \
             patch("cleancut.cli.build_edl", return_value=(_make_edl(), [])), \
             patch("cleancut.editor.probe_duration", return_value=None), \
             patch("cleancut.cli.write_report"), \
             patch("cleancut.cli.build_results_report", return_value=""):
            result = cmd_clean(args)

        assert result == 0
        assert captured["opts"].output == out

    def test_default_output_path_is_clean_suffix(self, tmp_path):
        args = _clean_args(tmp_path)
        video = Path(args.video)
        expected_default = video.with_name(f"{video.stem}.clean{video.suffix}")
        expected_default.write_bytes(b"\x00")
        captured = {}

        def fake_run(opts, config):
            captured["opts"] = opts
            return expected_default, _make_edl()

        with patch("cleancut.cli.run_full", side_effect=fake_run), \
             patch("cleancut.cli.build_edl", return_value=(_make_edl(), [])), \
             patch("cleancut.editor.probe_duration", return_value=None), \
             patch("cleancut.cli.write_report"), \
             patch("cleancut.cli.build_results_report", return_value=""):
            cmd_clean(args)

        assert captured["opts"].output == expected_default


# ---------------------------------------------------------------------------
# cmd_inspect
# ---------------------------------------------------------------------------

class TestCmdInspect:
    def _make_stream(self, **kw):
        from cleancut.probe import Stream
        defaults = dict(index=0, codec_name="aac", codec_type="audio",
                        language="eng", title="", channels=2)
        defaults.update(kw)
        return Stream(**defaults)

    def test_inspect_prints_plan(self, tmp_path, capsys):
        args = _inspect_args(tmp_path)
        fake_streams = [self._make_stream()]

        with patch("cleancut.cli.probe_streams", return_value=fake_streams), \
             patch("cleancut.cli.audio_streams", return_value=fake_streams), \
             patch("cleancut.cli.subtitle_streams", return_value=[]), \
             patch("cleancut.cli.pick_embedded_subtitle", return_value=None), \
             patch("cleancut.cli.find_sidecar_subtitle", return_value=None), \
             patch("cleancut.cli.build_plan", return_value="Mock plan text"):
            result = cmd_inspect(args)

        assert result == 0

    def test_inspect_missing_video_returns_1(self, tmp_path):
        args = _inspect_args(tmp_path)
        # Point to a path that doesn't exist
        args.video = str(tmp_path / "nonexistent.mp4")
        result = cmd_inspect(args)
        assert result == 1

    def test_inspect_calls_probe_streams(self, tmp_path):
        args = _inspect_args(tmp_path)
        fake_streams = []

        with patch("cleancut.cli.probe_streams", return_value=fake_streams) as mock_probe, \
             patch("cleancut.cli.audio_streams", return_value=[]), \
             patch("cleancut.cli.subtitle_streams", return_value=[]), \
             patch("cleancut.cli.pick_embedded_subtitle", return_value=None), \
             patch("cleancut.cli.find_sidecar_subtitle", return_value=None), \
             patch("cleancut.cli.build_plan", return_value=""):
            cmd_inspect(args)

        mock_probe.assert_called_once_with(Path(args.video))


# ---------------------------------------------------------------------------
# cmd_add_cut
# ---------------------------------------------------------------------------

class TestCmdAddCut:
    def _make_edl_file(self, tmp_path: Path, video_path: str = "/fake/video.mp4") -> Path:
        edl_path = tmp_path / "test.edl.json"
        edl = EditDecisionList(decisions=[], video_path=video_path)
        edl.to_json(edl_path)
        return edl_path

    def _add_cut_args(self, edl_path: Path, start="0:10", end="0:20", **extra):
        defaults = dict(
            edl=str(edl_path),
            start=start,
            end=end,
            action="cut",
            category="manual",
            reason=None,
            snap=False,
        )
        defaults.update(extra)
        return argparse.Namespace(**defaults)

    def test_adds_cut_to_edl(self, tmp_path):
        edl_path = self._make_edl_file(tmp_path)
        args = self._add_cut_args(edl_path, start="0:10", end="0:20")

        with patch("cleancut.editor.probe_duration", return_value=None), \
             patch("cleancut.cli.write_report"), \
             patch("cleancut.cli.build_results_report", return_value=""):
            result = cmd_add_cut(args)

        assert result == 0
        loaded = EditDecisionList.from_json(edl_path)
        assert len(loaded.decisions) == 1
        assert abs(loaded.decisions[0].start - 10.0) < 0.01
        assert abs(loaded.decisions[0].end - 20.0) < 0.01
        assert loaded.decisions[0].action == "cut"

    def test_end_before_start_returns_1(self, tmp_path):
        edl_path = self._make_edl_file(tmp_path)
        args = self._add_cut_args(edl_path, start="0:20", end="0:10")

        result = cmd_add_cut(args)
        assert result == 1

    def test_missing_edl_returns_1(self, tmp_path):
        args = self._add_cut_args(tmp_path / "nonexistent.edl.json")
        result = cmd_add_cut(args)
        assert result == 1

    def test_custom_category_and_reason(self, tmp_path):
        edl_path = self._make_edl_file(tmp_path)
        args = self._add_cut_args(
            edl_path, start="1:00", end="1:30",
            category="sex", reason="explicit scene"
        )

        with patch("cleancut.editor.probe_duration", return_value=None), \
             patch("cleancut.cli.write_report"), \
             patch("cleancut.cli.build_results_report", return_value=""):
            result = cmd_add_cut(args)

        assert result == 0
        loaded = EditDecisionList.from_json(edl_path)
        assert loaded.decisions[0].category == "sex"
        assert "explicit scene" in loaded.decisions[0].reason

    def test_mute_action_is_preserved(self, tmp_path):
        edl_path = self._make_edl_file(tmp_path)
        args = self._add_cut_args(edl_path, start="0:05", end="0:08", action="mute")

        with patch("cleancut.editor.probe_duration", return_value=None), \
             patch("cleancut.cli.write_report"), \
             patch("cleancut.cli.build_results_report", return_value=""):
            result = cmd_add_cut(args)

        assert result == 0
        loaded = EditDecisionList.from_json(edl_path)
        assert loaded.decisions[0].action == "mute"

    def test_multiple_cuts_are_sorted(self, tmp_path):
        edl_path = self._make_edl_file(tmp_path)
        # Add first cut at 30-40s
        args1 = self._add_cut_args(edl_path, start="0:30", end="0:40")
        with patch("cleancut.editor.probe_duration", return_value=None), \
             patch("cleancut.cli.write_report"), \
             patch("cleancut.cli.build_results_report", return_value=""):
            cmd_add_cut(args1)

        # Add second cut earlier at 5-10s
        args2 = self._add_cut_args(edl_path, start="0:05", end="0:10")
        with patch("cleancut.editor.probe_duration", return_value=None), \
             patch("cleancut.cli.write_report"), \
             patch("cleancut.cli.build_results_report", return_value=""):
            cmd_add_cut(args2)

        loaded = EditDecisionList.from_json(edl_path)
        assert len(loaded.decisions) == 2
        assert loaded.decisions[0].start < loaded.decisions[1].start


# ---------------------------------------------------------------------------
# cmd_review
# ---------------------------------------------------------------------------

class TestCmdReview:
    def _make_edl_file_with_cuts(self, tmp_path: Path) -> tuple[Path, Path]:
        video = tmp_path / "movie.mp4"
        video.write_bytes(b"\x00")
        edl_path = tmp_path / "movie.edl.json"
        edl = EditDecisionList(
            decisions=[
                EditDecision(
                    start=10.0, end=20.0, action="cut",
                    category="nudity", source="visual",
                    reason="test", accepted=True,
                ),
            ],
            video_path=str(video),
        )
        edl.to_json(edl_path)
        return edl_path, video

    def _review_args(self, edl_path: Path, video: Path | None = None, **extra):
        from cleancut.cli import build_parser

        argv = ["review", str(edl_path)]
        if video:
            argv += ["--video", str(video)]
        ns = build_parser().parse_args(argv)
        for k, v in extra.items():
            setattr(ns, k, v)
        return ns

    def test_missing_edl_returns_1(self, tmp_path):
        args = self._review_args(tmp_path / "nonexistent.edl.json",
                                 video=tmp_path / "video.mp4")
        result = cmd_review(args)
        assert result == 1

    def test_missing_video_returns_1(self, tmp_path):
        edl_path, _ = self._make_edl_file_with_cuts(tmp_path)
        args = self._review_args(edl_path, video=tmp_path / "does_not_exist.mp4")
        result = cmd_review(args)
        assert result == 1

    def test_no_cuts_to_review_returns_0(self, tmp_path):
        """EDL with no accepted cuts prints a message and returns 0."""
        video = tmp_path / "movie.mp4"
        video.write_bytes(b"\x00")
        edl_path = tmp_path / "no_cuts.edl.json"
        edl = EditDecisionList(decisions=[], video_path=str(video))
        edl.to_json(edl_path)

        args = self._review_args(edl_path, video=video)
        result = cmd_review(args)
        assert result == 0

    def test_review_accept_cut_with_y(self, tmp_path):
        """Entering 'y' keeps the cut accepted."""
        edl_path, video = self._make_edl_file_with_cuts(tmp_path)
        args = self._review_args(edl_path, video=video)

        with patch("subprocess.run"), \
             patch("builtins.input", return_value="y"), \
             patch("cleancut.editor.probe_duration", return_value=None), \
             patch("cleancut.cli.write_report"), \
             patch("cleancut.cli.build_results_report", return_value=""):
            result = cmd_review(args)

        assert result == 0
        loaded = EditDecisionList.from_json(edl_path)
        assert loaded.decisions[0].accepted is True

    def test_review_reject_cut_with_n(self, tmp_path):
        """Entering 'n' marks the cut as not accepted."""
        edl_path, video = self._make_edl_file_with_cuts(tmp_path)
        args = self._review_args(edl_path, video=video)

        with patch("subprocess.run"), \
             patch("builtins.input", return_value="n"), \
             patch("cleancut.editor.probe_duration", return_value=None), \
             patch("cleancut.cli.write_report"), \
             patch("cleancut.cli.build_results_report", return_value=""):
            result = cmd_review(args)

        assert result == 0
        loaded = EditDecisionList.from_json(edl_path)
        assert loaded.decisions[0].accepted is False

    def test_review_quit_early_with_q(self, tmp_path):
        """Entering 'q' saves and exits immediately."""
        edl_path, video = self._make_edl_file_with_cuts(tmp_path)
        args = self._review_args(edl_path, video=video)

        with patch("subprocess.run"), \
             patch("builtins.input", return_value="q"), \
             patch("cleancut.editor.probe_duration", return_value=None), \
             patch("cleancut.cli.write_report"), \
             patch("cleancut.cli.build_results_report", return_value=""):
            result = cmd_review(args)

        assert result == 0
