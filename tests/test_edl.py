from pathlib import Path

from cleancut.edl import EditDecision, EditDecisionList


def _d(start, end, action="mute", category="profanity"):
    return EditDecision(start=start, end=end, action=action, category=category)


def test_merge_overlapping_basic():
    edl = EditDecisionList(decisions=[
        _d(1, 3),
        _d(2.5, 4),     # overlaps prev
        _d(10, 12),     # separate
    ])
    merged = edl.merge_overlapping()
    assert len(merged) == 2
    assert merged.decisions[0].start == 1
    assert merged.decisions[0].end == 4
    assert merged.decisions[1].start == 10


def test_merge_overlapping_gap():
    edl = EditDecisionList(decisions=[
        _d(1, 3),
        _d(3.3, 5),    # 0.3s gap
    ])
    merged = edl.merge_overlapping(gap=0.5)
    assert len(merged) == 1
    assert merged.decisions[0].end == 5


def test_cut_wins_over_mute():
    edl = EditDecisionList(decisions=[
        _d(1, 3, action="mute"),
        _d(2, 4, action="cut"),
    ])
    merged = edl.merge_overlapping()
    assert merged.decisions[0].action == "cut"


def test_roundtrip_json(tmp_path: Path):
    edl = EditDecisionList(
        decisions=[_d(1, 2), _d(5, 7, action="cut", category="nudity")],
        video_path="/tmp/video.mp4",
    )
    p = tmp_path / "edl.json"
    edl.to_json(p)
    loaded = EditDecisionList.from_json(p)
    assert loaded.video_path == "/tmp/video.mp4"
    assert len(loaded) == 2
    assert loaded.decisions[1].category == "nudity"
