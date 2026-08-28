"""Tests for behavior corrections that ride along with the dead-code cleanup."""
from __future__ import annotations

from unittest.mock import patch

from cleancut.edl import EditDecision, EditDecisionList
from cleancut.editor_ranges import Range, shift_after_cuts
from cleancut.subtitles import Subtitle


class TestMultiCategory:
    def test_llm_multi_is_labeled_multi_not_sex_drugs(self):
        """A drugs+violence scene must not be reported as 'sex+drugs'."""
        from cleancut.classify_dialogue import LLMParams, classify_dialogue

        subs = [Subtitle(index=1, start=0.0, end=2.0, text="a"),
                Subtitle(index=2, start=3.0, end=5.0, text="b")]

        class Client:
            def generate(self, **kw):
                return {"response": "ok"}

            def chat(self, **kw):
                return {"message": {"content":
                    '{"category": "multi", "should_cut": true, '
                    '"confidence": 0.9, "reasoning": "x"}'}}

        with patch("cleancut.classify_dialogue.make_ollama_client", return_value=Client()):
            edl = classify_dialogue(subs, LLMParams(), use_cache=False)
        assert len(edl) == 1
        assert edl.decisions[0].category == "multi"


class TestCutsForReview:
    def _edl(self):
        return EditDecisionList(decisions=[
            EditDecision(start=0, end=1, action="cut", category="sex"),
            EditDecision(start=2, end=3, action="cut", category="violence"),
            EditDecision(start=4, end=5, action="cut", category="multi"),
            EditDecision(start=6, end=7, action="cut", category="nudity", accepted=False),
            EditDecision(start=8, end=9, action="mute", category="profanity"),
        ])

    def test_default_hides_violence_keeps_multi(self):
        from cleancut.edl_ops import cuts_for_review

        cats = [d.category for d in cuts_for_review(self._edl())]
        assert cats == ["sex", "multi"]

    def test_include_violence(self):
        from cleancut.edl_ops import cuts_for_review

        cats = [d.category for d in cuts_for_review(self._edl(), include_violence=True)]
        assert cats == ["sex", "violence", "multi"]


class TestShiftAfterCuts:
    def test_t_at_cut_start_is_inside_the_cut(self):
        """t == cut.start is the first removed instant — must map to None,
        not pass through unshifted."""
        assert shift_after_cuts(10.0, [Range(10.0, 20.0)]) is None

    def test_overlapping_cuts_do_not_double_subtract(self):
        """Two overlapping cuts remove their union, not the sum of durations."""
        cuts = [Range(10.0, 20.0), Range(15.0, 25.0)]  # union removes 15s
        assert shift_after_cuts(30.0, cuts) == 15.0
