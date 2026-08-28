from cleancut.corroboration import _is_visual_only, mark_unsupported_visual
from cleancut.edl import EditDecision, EditDecisionList


def _d(s, e, source, accepted=True):
    return EditDecision(start=s, end=e, action="cut", category="nudity",
                        source=source, accepted=accepted)


def test_is_visual_only_pure_vlm():
    assert _is_visual_only("vlm") is True


def test_is_visual_only_visual_shot():
    assert _is_visual_only("visual-shot") is True


def test_is_visual_only_false_when_combined_with_dialogue():
    assert _is_visual_only("vlm+subtitle") is False
    assert _is_visual_only("visual-shot+llm-dialogue") is False


def test_is_visual_only_false_when_combined_with_audio():
    assert _is_visual_only("vlm+audio") is False


def test_is_visual_only_false_for_pure_dialogue():
    assert _is_visual_only("subtitle") is False
    assert _is_visual_only("llm-dialogue") is False


def test_solo_vlm_with_no_neighbors_marked_unaccepted():
    edl = EditDecisionList(decisions=[_d(100, 105, "vlm")])
    edl, n = mark_unsupported_visual(edl, radius_seconds=5)
    assert n == 1
    assert edl.decisions[0].accepted is False


def test_solo_vlm_with_dialogue_neighbor_kept():
    edl = EditDecisionList(decisions=[
        _d(100, 105, "vlm"),
        _d(108, 110, "subtitle"),     # within radius
    ])
    edl, n = mark_unsupported_visual(edl, radius_seconds=5)
    assert n == 0
    assert edl.decisions[0].accepted is True


def test_solo_vlm_with_neighbor_outside_radius_dropped():
    edl = EditDecisionList(decisions=[
        _d(100, 105, "vlm"),
        _d(200, 205, "subtitle"),     # too far
    ])
    edl, n = mark_unsupported_visual(edl, radius_seconds=5)
    assert n == 1
    assert edl.decisions[0].accepted is False


def test_corroborator_must_be_accepted():
    edl = EditDecisionList(decisions=[
        _d(100, 105, "vlm"),
        _d(106, 108, "subtitle", accepted=False),    # rejected — doesn't corroborate
    ])
    edl, n = mark_unsupported_visual(edl, radius_seconds=10)
    assert n == 1


def test_already_corroborated_combined_source_kept():
    edl = EditDecisionList(decisions=[_d(100, 105, "vlm+subtitle")])
    edl, n = mark_unsupported_visual(edl, radius_seconds=5)
    assert n == 0
    assert edl.decisions[0].accepted is True


def test_audio_event_corroborates_visual():
    edl = EditDecisionList(decisions=[
        _d(100, 105, "vlm"),
        EditDecision(start=103, end=110, action="cut",
                     category="sex", source="audio", accepted=True),
    ])
    edl, n = mark_unsupported_visual(edl, radius_seconds=2)
    assert n == 0


def test_density_event_corroborates_visual():
    edl = EditDecisionList(decisions=[
        _d(100, 105, "vlm"),
        EditDecision(start=102, end=108, action="cut",
                     category="drugs", source="density", accepted=True),
    ])
    edl, n = mark_unsupported_visual(edl, radius_seconds=2)
    assert n == 0


def test_reason_annotated_when_marked():
    edl = EditDecisionList(decisions=[_d(100, 105, "vlm")])
    edl.decisions[0].reason = "VLM (llava:7b) drugs: cigarette"
    edl, _ = mark_unsupported_visual(edl, radius_seconds=5)
    assert "solo-visual" in edl.decisions[0].reason
