"""Tests for EditDecisionList methods not covered by test_edl."""

from cleancut.edl import EditDecision, EditDecisionList


def _d(s, e, action="mute", category="profanity", accepted=True):
    return EditDecision(start=s, end=e, action=action, category=category, accepted=accepted)


def test_pad_extends_both_sides():
    edl = EditDecisionList(decisions=[_d(10, 20)])
    padded = edl.pad(1.5)
    assert padded.decisions[0].start == 8.5
    assert padded.decisions[0].end == 21.5


def test_pad_clamps_start_at_zero():
    edl = EditDecisionList(decisions=[_d(0.5, 5)])
    padded = edl.pad(2)
    assert padded.decisions[0].start == 0.0


def test_summary_counts_by_action_category():
    edl = EditDecisionList(decisions=[
        _d(0, 1, action="mute", category="profanity"),
        _d(2, 3, action="mute", category="profanity"),
        _d(4, 5, action="cut", category="sex"),
    ])
    summary = edl.summary()
    assert summary == {"mute:profanity": 2, "cut:sex": 1}


def test_summary_strips_combined_categories():
    edl = EditDecisionList(decisions=[
        _d(0, 1, action="cut", category="sex+drugs"),
    ])
    summary = edl.summary()
    # Should bucket under the first category only
    assert summary == {"cut:sex": 1}


def test_by_action_only_returns_accepted():
    edl = EditDecisionList(decisions=[
        _d(0, 1, action="mute", accepted=True),
        _d(2, 3, action="mute", accepted=False),
        _d(4, 5, action="cut", accepted=True),
    ])
    mutes = edl.by_action("mute")
    assert len(mutes) == 1
    cuts = edl.by_action("cut")
    assert len(cuts) == 1


def test_len_and_iter():
    edl = EditDecisionList(decisions=[_d(0, 1), _d(2, 3)])
    assert len(edl) == 2
    assert sum(1 for _ in edl) == 2


def test_sorted_returns_new_list():
    edl = EditDecisionList(decisions=[_d(5, 6), _d(0, 1)])
    out = edl.sorted()
    assert [d.start for d in out] == [0, 5]
    # Original unchanged
    assert edl.decisions[0].start == 5


def test_duration_property():
    assert _d(10, 15).duration == 5
    assert _d(20, 18).duration == 0.0


def test_merge_overlapping_collapses_source_tags():
    edl = EditDecisionList(decisions=[
        EditDecision(start=0, end=5, action="mute", category="profanity", source="subtitle"),
        EditDecision(start=4, end=8, action="mute", category="profanity", source="density"),
    ])
    merged = edl.merge_overlapping()
    assert len(merged) == 1
    assert "subtitle" in merged.decisions[0].source
    assert "density" in merged.decisions[0].source
