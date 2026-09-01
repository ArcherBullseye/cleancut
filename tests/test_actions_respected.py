"""Detectors must honour the user's per-category actions.

classify_dialogue, classify_visual, audio_events and density all hardcoded
action="cut", so a scan run with violence=keep still cut every violent scene the
LLM found -- and a cut, unlike a mute, forces a full video re-encode.
"""
from __future__ import annotations

import pytest

from cleancut.density import DensityParams, find_clusters
from cleancut.edl import EditDecision, EditDecisionList, resolve_action


class TestResolveAction:
    def test_no_actions_configured_defaults_to_cut(self):
        assert resolve_action("violence", None) == "cut"
        assert resolve_action("violence", {}) == "cut"

    def test_configured_category_is_honoured(self):
        assert resolve_action("violence", {"violence": "keep"}) == "keep"
        assert resolve_action("sex", {"sex": "mute"}) == "mute"

    @pytest.mark.parametrize(
        "actions,expected",
        [
            ({"violence": "keep", "profanity": "mute"}, "mute"),
            ({"violence": "keep", "sex": "cut"}, "cut"),
            ({"violence": "keep", "profanity": "keep"}, "keep"),
            ({"violence": "mute", "sex": "cut"}, "cut"),
        ],
    )
    def test_composite_takes_the_strongest_component(self, actions, expected):
        """A composite category is only dropped if every component says keep."""
        category = "+".join(sorted(actions))
        assert resolve_action(category, actions) == expected

    def test_unknown_category_falls_back_to_default(self):
        """The LLM's catch-all "multi" has no per-category setting."""
        assert resolve_action("multi", {"violence": "keep"}) == "cut"
        assert resolve_action("multi", {"violence": "keep"}, default="mute") == "mute"


def _events(category: str, source: str = "subtitle") -> EditDecisionList:
    edl = EditDecisionList()
    for start in (0.0, 5.0, 10.0):
        edl.add(EditDecision(start=start, end=start + 4.0, action="mute",
                             category=category, reason="x", source=source))
    return edl


class TestDensityRespectsActions:
    params = DensityParams(window_seconds=60.0, min_events=3, min_cluster_span=8.0)

    def test_cluster_is_cut_when_unconfigured(self):
        out = find_clusters(_events("profanity"), self.params)
        assert [d.action for d in out.decisions] == ["cut"]

    def test_cluster_is_muted_when_category_is_muted(self):
        params = DensityParams(**{**vars(self.params), "actions": {"profanity": "mute"}})
        out = find_clusters(_events("profanity"), params)
        assert [d.action for d in out.decisions] == ["mute"]

    def test_cluster_is_dropped_when_category_is_kept(self):
        """The loop must still advance -- a `continue` here would hang."""
        params = DensityParams(**{**vars(self.params), "actions": {"violence": "keep"}})
        out = find_clusters(_events("violence"), params)
        assert list(out.decisions) == []


class TestDetectorParamsAcceptActions:
    """Every detector that emits decisions must be able to receive the config."""

    def test_all_four_params_carry_actions(self):
        from cleancut.audio_events import AudioEventParams
        from cleancut.classify_dialogue import LLMParams
        from cleancut.classify_visual import VLMParams

        for cls in (LLMParams, VLMParams, AudioEventParams, DensityParams):
            assert cls().actions is None, cls.__name__
            assert cls(actions={"sex": "mute"}).actions == {"sex": "mute"}, cls.__name__


class TestPipelineWiresActionsThrough:
    def test_every_detector_construction_passes_actions(self):
        """A detector that respects actions but is never handed them is no fix."""
        from pathlib import Path

        src = (Path(__file__).resolve().parent.parent / "cleancut" / "pipeline.py").read_text()
        assert src.count("actions=config.actions") == 4, (
            "expected all four detector params to receive config.actions"
        )
