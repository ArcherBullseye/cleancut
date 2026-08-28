"""Tests for subtitles.py: read/write SRT roundtrip, soften_text edge cases."""

from pathlib import Path

from cleancut.subtitles import Subtitle, read_srt, soften_text, write_srt


def test_write_then_read_roundtrip(tmp_path: Path):
    subs = [
        Subtitle(index=1, start=1.0, end=3.5, text="Hello"),
        Subtitle(index=2, start=5.25, end=7.0, text="World"),
    ]
    p = tmp_path / "x.srt"
    write_srt(subs, p)
    loaded = read_srt(p)
    assert len(loaded) == 2
    assert loaded[0].text == "Hello"
    assert loaded[0].start == 1.0
    assert loaded[1].end == 7.0


def test_soften_text_empty_replacements():
    assert soften_text("anything goes", {}) == "anything goes"


def test_soften_text_no_match():
    assert soften_text("perfectly clean", {"fuck": "freak"}) == "perfectly clean"


def test_soften_text_preserves_punctuation():
    out = soften_text("Fuck!", {"fuck": "freak"})
    assert out == "Freak!"


def test_soften_text_multi_word_preferred_over_single():
    # Longer phrases first.
    repl = {"sleep with": "date", "sleep": "rest"}
    out = soften_text("I want to sleep with you", repl)
    assert "date" in out
    # The remaining "sleep" should NOT trigger because the phrase consumed it
    assert "rest" not in out


def test_soften_text_case_all_upper():
    assert soften_text("FUCK IT", {"fuck": "freak"}) == "FREAK IT"


def test_soften_text_case_title():
    assert soften_text("Fuck no", {"fuck": "freak"}) == "Freak no"


def test_read_srt_handles_utf8(tmp_path: Path):
    p = tmp_path / "x.srt"
    p.write_text("1\n00:00:00,000 --> 00:00:01,000\nCafé naïve\n", encoding="utf-8")
    subs = read_srt(p)
    assert subs[0].text == "Café naïve"
