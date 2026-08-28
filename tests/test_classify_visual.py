from cleancut.classify_visual import VLMParams, _flagged_categories, select_shots
from cleancut.edl import EditDecision, EditDecisionList
from cleancut.scenes import Shot
from cleancut.subtitles import Subtitle


def _shot(s, e):
    return Shot(s, e)


def _sub(idx, s, e, t="line"):
    return Subtitle(index=idx, start=s, end=e, text=t)


def test_select_shots_silent_mode_excludes_shots_with_dialogue():
    shots = [_shot(0, 5), _shot(5, 10), _shot(10, 15)]
    subs = [_sub(1, 6, 9, "talk")]
    out = select_shots(shots, subs, EditDecisionList(), VLMParams(mode="silent"))
    times = [(s.start, s.end) for s in out]
    assert (0, 5) in times
    assert (10, 15) in times
    assert (5, 10) not in times


def test_select_shots_gaps_mode_returns_neighbors_of_flagged():
    shots = [_shot(0, 10), _shot(10, 20), _shot(20, 30), _shot(100, 110)]
    flagged = EditDecisionList(decisions=[
        EditDecision(start=12, end=14, action="cut", category="sex")
    ])
    out = select_shots(shots, [], flagged, VLMParams(mode="gaps", gaps_radius_seconds=15))
    times = [(s.start, s.end) for s in out]
    assert (0, 10) in times
    assert (10, 20) in times
    assert (20, 30) in times
    assert (100, 110) not in times


def test_select_shots_composite_mode():
    # silent+gaps: include either silent shots OR adjacent to flagged
    shots = [_shot(0, 10), _shot(10, 20), _shot(20, 30), _shot(50, 60)]
    subs = [_sub(1, 11, 19)]  # dialogue covers shot 10-20
    flagged = EditDecisionList(decisions=[
        EditDecision(start=55, end=58, action="cut", category="sex")
    ])
    out = select_shots(shots, subs, flagged, VLMParams(mode="silent+gaps", gaps_radius_seconds=10))
    times = [(s.start, s.end) for s in out]
    # shot 0-10 is silent
    assert (0, 10) in times
    # shot 50-60 is adjacent to flagged
    assert (50, 60) in times


def test_flagged_categories_respects_confidence():
    result = {"explicit": True, "confidence": 0.4}
    assert _flagged_categories(result, VLMParams(min_confidence=0.55)) == []
    result["confidence"] = 0.9
    assert _flagged_categories(result, VLMParams(min_confidence=0.55)) == ["nudity"]


def test_flagged_categories_intimate_only_if_opted_in():
    result = {"intimate": True, "confidence": 0.9}
    # default cut_on excludes intimate
    assert _flagged_categories(result, VLMParams()) == []
    # opt-in
    p = VLMParams(cut_on=("intimate", "explicit", "drug_use", "violence"))
    assert _flagged_categories(result, p) == ["sex"]
