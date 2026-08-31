"""The subtitle softener must cover the same words the detector flags.

The detector spells open suffixes (\\w*) in its wordlists, so it matches "porno",
"fuckers", "raped". The replacements map holds base forms only, so those words
were detected -- muted or cut in the audio -- while the softened subtitle track
still displayed them in full.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from cleancut.subtitles import soften_text

DATA = Path(__file__).resolve().parent.parent / "cleancut" / "data"


@pytest.fixture(scope="module")
def reps() -> dict:
    return json.loads((DATA / "replacements.json").read_text())


@pytest.fixture(scope="module")
def words() -> dict:
    return json.loads((DATA / "wordlists.json").read_text())


class TestInflectedFormsAreSoftened:
    @pytest.mark.parametrize("text", ["porno", "pornography", "fuckers", "raped", "bitches", "damned", "crappy"])
    def test_inflection_no_longer_survives(self, text, reps, words):
        assert soften_text(text, reps, words).lower() != text

    def test_the_line_that_started_this(self, reps, words):
        line = "I was just gonna send a thousand porno magazines to his office,"
        assert "porno" not in soften_text(line, reps, words).lower()


class TestInnocentWordsAreUntouched:
    """`ass` is a replacement key. Suffixing every key would turn "assistant"
    into nonsense; only the detector's own patterns may widen a match, and it
    has no bare-`ass` pattern."""

    @pytest.mark.parametrize(
        "text",
        ["my assistant will assume the assignment", "he was assessing it",
         "that is classic", "a niggling doubt", "the class assembled"],
    )
    def test_unchanged(self, text, reps, words):
        assert soften_text(text, reps, words) == text


class TestNoBaseFormIsLeftAlone:
    """An inflection the map cannot resolve is returned as-is. Readable beats
    mangled, and the audio is still muted or cut either way."""

    @pytest.mark.parametrize("text", ["tortured", "molested"])
    def test_untouched_when_unresolvable(self, text, reps, words):
        assert soften_text(text, reps, words) == text


class TestBackwardCompatible:
    def test_omitting_wordlists_matches_old_behaviour(self, reps):
        assert soften_text("what the hell", reps) == "what the heck"
        assert soften_text("porno", reps) == "porno"

    def test_case_is_preserved(self, reps, words):
        assert soften_text("PORNO", reps, words).isupper()
        assert soften_text("Porno", reps, words)[0].isupper()

    def test_empty_replacements_is_a_noop(self, words):
        assert soften_text("porno", {}, words) == "porno"
