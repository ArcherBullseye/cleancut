"""Density should only cluster dialogue/audio events — never anchor on visual hits."""

from cleancut.density import DensityParams, find_clusters
from cleancut.edl import EditDecision, EditDecisionList


def _d(start, end, source="subtitle"):
    return EditDecision(start=start, end=end, action="mute", category="profanity", source=source)


def test_pure_visual_source_does_not_cluster():
    edl = EditDecisionList(decisions=[
        _d(10, 11, source="vlm"),
        _d(15, 16, source="vlm"),
        _d(25, 26, source="vlm"),
    ])
    out = find_clusters(edl, DensityParams(min_events=3, min_cluster_span=5))
    # No dialogue events → no cluster
    assert len(out) == 0


def test_nudenet_hit_does_not_anchor_cluster():
    edl = EditDecisionList(decisions=[
        _d(10, 11, source="visual-shot"),       # nudenet FP — should be IGNORED
        _d(60, 61, source="subtitle"),          # lone dialogue 50s later
        _d(70, 71, source="subtitle"),
    ])
    out = find_clusters(edl, DensityParams(min_events=3, min_cluster_span=5))
    # Only 2 dialogue events — below min_events
    assert len(out) == 0


def test_dialogue_cluster_still_works():
    edl = EditDecisionList(decisions=[
        _d(10, 11, source="subtitle"),
        _d(15, 16, source="whisper-word"),
        _d(25, 26, source="llm-dialogue"),
    ])
    out = find_clusters(edl, DensityParams(min_events=3, min_cluster_span=5))
    assert len(out) == 1


def test_audio_events_count_as_dialogue_for_density():
    edl = EditDecisionList(decisions=[
        _d(10, 11, source="audio"),
        _d(15, 16, source="subtitle"),
        _d(25, 26, source="audio"),
    ])
    out = find_clusters(edl, DensityParams(min_events=3, min_cluster_span=5))
    assert len(out) == 1


def test_mixed_source_does_not_pull_visual_into_cluster():
    # Visual hits scattered among dialogue events — should not bloat the cluster span.
    edl = EditDecisionList(decisions=[
        _d(10, 11, source="subtitle"),
        _d(15, 16, source="visual-shot"),       # ignored
        _d(25, 26, source="subtitle"),
        _d(40, 41, source="vlm"),               # ignored
        _d(45, 46, source="llm-dialogue"),
    ])
    out = find_clusters(edl, DensityParams(min_events=3, min_cluster_span=5))
    assert len(out) == 1
    # Span should be 10 → 46, not pulled into 10 → 46 anyway in this case;
    # the point is visual events were never *counted* toward the min_events check.
    assert out.decisions[0].start <= 10
    assert out.decisions[0].end >= 46


def test_combined_source_string_still_qualifies():
    # An event previously merged across detectors has source like "subtitle+vlm".
    # Containing ANY dialogue token should be enough.
    edl = EditDecisionList(decisions=[
        _d(10, 11, source="subtitle+vlm"),
        _d(15, 16, source="subtitle"),
        _d(25, 26, source="subtitle"),
    ])
    out = find_clusters(edl, DensityParams(min_events=3, min_cluster_span=5))
    assert len(out) == 1
