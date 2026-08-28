"""Tests for editor.py ffmpeg command construction and render orchestration.

subprocess.run is mocked throughout — these tests assert on the command
lists cleancut builds, not on ffmpeg behavior.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from cleancut.edl import EditDecision, EditDecisionList
from cleancut.editor import apply_mutes_and_subs


def _capture_run():
    """Return (calls, fake_run) where calls collects (cmd, kwargs)."""
    calls: list[tuple[list[str], dict]] = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))

        class R:
            returncode = 0

        return R()

    return calls, fake_run


class TestApplyMutesAndSubs:
    def test_burn_path_resolves_relative_paths_to_absolute(self, tmp_path, monkeypatch):
        """With cwd=safe_dir, relative input/output paths would resolve inside the
        temp dir (output then destroyed by cleanup). Paths must be absolute in cmd."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "in.mp4").write_bytes(b"\x00")
        (tmp_path / "subs.srt").write_text("1\n00:00:01,000 --> 00:00:02,000\nhi\n")

        calls, fake_run = _capture_run()
        with patch("cleancut.editor.subprocess.run", side_effect=fake_run), \
             patch("cleancut.editor._ffmpeg_has_libass", return_value=True), \
             patch("cleancut.editor._require_ffmpeg"):
            apply_mutes_and_subs(
                input_path=Path("in.mp4"),
                mutes=[],
                srt_path=Path("subs.srt"),
                output_path=Path("out.mp4"),
                burn_subs=True,
            )

        cmd, kwargs = calls[0]
        assert kwargs.get("cwd") is not None  # burn mode runs in a temp dir
        input_arg = cmd[cmd.index("-i") + 1]
        output_arg = cmd[-1]
        assert Path(input_arg).is_absolute(), f"input not absolute: {input_arg}"
        assert Path(output_arg).is_absolute(), f"output not absolute: {output_arg}"

    def test_soft_subs_mkv_uses_srt_codec(self, tmp_path):
        """mov_text is MP4-only; muxing it into Matroska fails. .mkv output
        must use the srt subtitle codec."""
        srt = tmp_path / "subs.srt"
        srt.write_text("1\n00:00:01,000 --> 00:00:02,000\nhi\n")

        calls, fake_run = _capture_run()
        with patch("cleancut.editor.subprocess.run", side_effect=fake_run), \
             patch("cleancut.editor._ffmpeg_has_libass", return_value=False), \
             patch("cleancut.editor._require_ffmpeg"):
            apply_mutes_and_subs(
                input_path=tmp_path / "in.mkv",
                mutes=[],
                srt_path=srt,
                output_path=tmp_path / "out.mkv",
                burn_subs=False,
            )

        cmd, _ = calls[0]
        assert "mov_text" not in cmd
        assert cmd[cmd.index("-c:s") + 1] == "srt"

    def test_soft_subs_mp4_uses_mov_text(self, tmp_path):
        srt = tmp_path / "subs.srt"
        srt.write_text("1\n00:00:01,000 --> 00:00:02,000\nhi\n")

        calls, fake_run = _capture_run()
        with patch("cleancut.editor.subprocess.run", side_effect=fake_run), \
             patch("cleancut.editor._ffmpeg_has_libass", return_value=False), \
             patch("cleancut.editor._require_ffmpeg"):
            apply_mutes_and_subs(
                input_path=tmp_path / "in.mp4",
                mutes=[],
                srt_path=srt,
                output_path=tmp_path / "out.mp4",
                burn_subs=False,
            )

        cmd, _ = calls[0]
        assert cmd[cmd.index("-c:s") + 1] == "mov_text"


class TestApplyCuts:
    def test_no_cuts_remuxes_with_stream_copy(self, tmp_path):
        from cleancut.editor import apply_cuts

        calls, fake_run = _capture_run()
        with patch("cleancut.editor.subprocess.run", side_effect=fake_run), \
             patch("cleancut.editor._require_ffmpeg"):
            apply_cuts(tmp_path / "in.mp4", [], tmp_path / "out.mp4")

        cmd, _ = calls[0]
        assert cmd[cmd.index("-c") + 1] == "copy"
        assert "-filter_complex" not in cmd

    def test_cuts_build_trim_concat_filter(self, tmp_path):
        from cleancut.editor import Range, apply_cuts

        calls, fake_run = _capture_run()
        with patch("cleancut.editor.subprocess.run", side_effect=fake_run), \
             patch("cleancut.editor.probe_duration", return_value=100.0), \
             patch("cleancut.editor._require_ffmpeg"):
            apply_cuts(tmp_path / "in.mp4", [Range(10.0, 20.0)], tmp_path / "out.mp4",
                       encoder="libx264", quality=20)

        cmd, _ = calls[0]
        fc = cmd[cmd.index("-filter_complex") + 1]
        # Two kept segments: [0,10] and [20,100], concatenated.
        assert "trim=start=0.000:end=10.000" in fc
        assert "trim=start=20.000:end=100.000" in fc
        assert "concat=n=2:v=1:a=1" in fc
        assert cmd[cmd.index("-c:v") + 1] == "libx264"

    def test_all_content_cut_raises(self, tmp_path):
        import pytest

        from cleancut.editor import Range, apply_cuts

        with patch("cleancut.editor.subprocess.run"), \
             patch("cleancut.editor.probe_duration", return_value=10.0), \
             patch("cleancut.editor._require_ffmpeg"):
            with pytest.raises(RuntimeError, match="[Nn]othing left"):
                apply_cuts(tmp_path / "in.mp4", [Range(0.0, 10.0)], tmp_path / "out.mp4")

    def test_mutes_render_volume_filter(self, tmp_path):
        calls, fake_run = _capture_run()
        with patch("cleancut.editor.subprocess.run", side_effect=fake_run), \
             patch("cleancut.editor._ffmpeg_has_libass", return_value=False), \
             patch("cleancut.editor._require_ffmpeg"):
            from cleancut.editor import Range
            apply_mutes_and_subs(
                input_path=tmp_path / "in.mp4",
                mutes=[Range(1.0, 2.5), Range(7.0, 8.0)],
                srt_path=None,
                output_path=tmp_path / "out.mp4",
                burn_subs=False,
            )

        cmd, _ = calls[0]
        af = cmd[cmd.index("-af") + 1]
        assert "between(t,1.000,2.500)" in af
        assert "between(t,7.000,8.000)" in af
        assert af.startswith("volume=enable=")


class TestRenderWorkDirCleanup:
    def test_render_removes_work_dir(self, tmp_path):
        from cleancut.config import Config
        from cleancut.pipeline import PipelineOptions, render

        video = tmp_path / "movie.mp4"
        video.write_bytes(b"\x00")
        out = tmp_path / "movie.clean.mp4"
        edl = EditDecisionList(decisions=[
            EditDecision(start=1.0, end=2.0, action="cut", category="nudity",
                         source="visual", reason="test"),
        ])
        opts = PipelineOptions(video=video, output=out, burn_subs=False)

        with patch("cleancut.pipeline.apply_cuts"), \
             patch("cleancut.pipeline.apply_mutes_and_subs"):
            render(edl, [], opts, Config.load_defaults())

        assert not (tmp_path / ".cleancut_work").exists()


class TestRunFullReturnsEdl:
    def test_run_full_returns_path_and_edl(self, tmp_path):
        """run_full must hand back the EDL it built so callers (cmd_clean's
        report step) never re-run the detection stack."""
        from cleancut.config import Config
        from cleancut.pipeline import PipelineOptions, run_full

        video = tmp_path / "movie.mp4"
        video.write_bytes(b"\x00")
        fake_edl = EditDecisionList(video_path=str(video))
        opts = PipelineOptions(video=video, edl_out=tmp_path / "movie.edl.json")

        with patch("cleancut.pipeline.build_edl", return_value=(fake_edl, [])):
            result = run_full(opts, Config.load_defaults())

        out_path, edl = result
        assert out_path == opts.edl_out
        assert edl is fake_edl
