"""Smoke tests for the CLI — verify every subcommand's argparse works."""

import pytest

from cleancut.cli import main


def test_help_top_level(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "cleancut" in out
    for sub in ("scan", "clean", "inspect", "add-cut", "review"):
        assert sub in out


@pytest.mark.parametrize("subcmd", ["scan", "clean", "inspect", "add-cut", "review"])
def test_subcommand_help_parses(subcmd, capsys):
    with pytest.raises(SystemExit) as exc:
        main([subcmd, "--help"])
    assert exc.value.code == 0


def test_missing_video_argument_for_scan_fails(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["scan"])
    # argparse exits 2 on missing required positional
    assert exc.value.code == 2


def test_invalid_action_format_for_clean_fails(tmp_path, capsys):
    # Use a non-existent file; we expect early arg-validation failure for --action
    fake = tmp_path / "x.mp4"
    fake.write_bytes(b"\x00")
    with pytest.raises(SystemExit):
        # --action without the expected CATEGORY=ACTION form
        main(["clean", str(fake), "--action", "no-equals-sign", "-o", str(tmp_path / "o.mp4")])


def test_invalid_preset_fails(tmp_path, capsys):
    fake = tmp_path / "x.mp4"
    fake.write_bytes(b"\x00")
    with pytest.raises(SystemExit):
        main(["scan", str(fake), "--preset", "made-up-preset"])


def test_add_cut_requires_start_and_end(tmp_path):
    edl = tmp_path / "e.json"
    edl.write_text('{"video_path":"/x.mp4","decisions":[]}')
    with pytest.raises(SystemExit) as exc:
        main(["add-cut", str(edl)])  # missing --start / --end
    assert exc.value.code == 2
