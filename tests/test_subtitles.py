from pathlib import Path

from cleancut.config import Config
from cleancut.subtitles import read_srt, scan_subtitles, soften_text

FIXTURE = Path(__file__).parent / "fixtures" / "sample.srt"


def test_read_srt():
    subs = read_srt(FIXTURE)
    assert len(subs) == 5
    assert subs[0].text == "Hello there, friend."
    assert subs[1].start == 4.0
    assert subs[1].end == 6.5


def test_scan_subtitles_matches_profanity_and_drugs():
    config = Config.load_defaults()
    subs = read_srt(FIXTURE)
    edl = scan_subtitles(subs, config)

    # Should flag the fuck line, the cocaine line.
    categories = {d.category for d in edl.decisions}
    assert "profanity" in categories
    assert "drugs" in categories

    # The benign line should not be flagged.
    flagged_texts = {d.text_before for d in edl.decisions}
    assert "Just a normal line of dialogue." not in flagged_texts


def test_soften_text_preserves_case():
    repl = {"fuck": "freak", "cocaine": "the stuff"}
    assert soften_text("What the fuck is going on?", repl) == "What the freak is going on?"
    assert soften_text("FUCK that.", repl) == "FREAK that."
    assert soften_text("Fuck.", repl) == "Freak."


def test_soften_text_word_boundary():
    repl = {"ass": "butt"}
    # Should match "ass" as a word but not "class" or "pass".
    assert soften_text("Don't be an ass.", repl) == "Don't be an butt."
    assert soften_text("First class flight.", repl) == "First class flight."
