from pathlib import Path

from cleancut.config import Config
from cleancut.subtitles import read_srt, scan_subtitles, scan_words, soften_text
from cleancut.transcribe import Word

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


def test_scan_words_mutes_only_the_matched_word_timestamp():
    config = Config.load_defaults()
    words = [
        Word(start=1.0, end=1.25, text="What"),
        Word(start=1.3, end=1.45, text="the"),
        Word(start=1.5, end=1.82, text="fuck"),
        Word(start=1.9, end=2.2, text="happened"),
    ]

    edl = scan_words(words, config)

    assert len(edl.decisions) == 1
    assert edl.decisions[0].category == "profanity"
    assert (edl.decisions[0].start, edl.decisions[0].end) == (1.5, 1.82)


def test_scan_words_keeps_exact_multiword_phrase_timestamp():
    config = Config.load_defaults()
    words = [
        Word(start=4.0, end=4.25, text="I"),
        Word(start=4.3, end=4.62, text="said"),
        Word(start=4.7, end=5.02, text="fuck,"),
        Word(start=5.08, end=5.3, text="you!"),
        Word(start=5.4, end=5.75, text="Leave"),
    ]

    edl = scan_words(words, config)

    assert len(edl.decisions) == 1
    assert edl.decisions[0].category == "sex"
    assert (edl.decisions[0].start, edl.decisions[0].end) == (4.7, 5.3)


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
