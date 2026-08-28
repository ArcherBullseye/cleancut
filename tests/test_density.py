from cleancut.density import DensityParams, find_clusters
from cleancut.edl import EditDecision, EditDecisionList


def _d(start, end, cat="profanity"):
    # Use "subtitle" source so events qualify as dialogue for density clustering.
    return EditDecision(start=start, end=end, action="mute", category=cat, source="subtitle")


def test_no_cluster_below_min_events():
    edl = EditDecisionList(decisions=[_d(10, 11), _d(20, 21)])
    out = find_clusters(edl, DensityParams(min_events=3))
    assert len(out) == 0


def test_dense_cluster_detected():
    # 4 events all within 30s — should fire one cut covering ~30s span.
    edl = EditDecisionList(decisions=[_d(10, 11), _d(15, 16), _d(25, 26), _d(40, 41)])
    out = find_clusters(edl, DensityParams(window_seconds=60, min_events=3, min_cluster_span=5))
    assert len(out) == 1
    assert out.decisions[0].action == "cut"
    assert out.decisions[0].start <= 10
    assert out.decisions[0].end >= 41


def test_two_separate_clusters():
    # Cluster A: 10-40s, Cluster B: 200-230s. No overlap.
    edl = EditDecisionList(decisions=[
        _d(10, 11), _d(20, 21), _d(40, 41),
        _d(200, 201), _d(210, 211), _d(230, 231),
    ])
    out = find_clusters(edl, DensityParams(window_seconds=60, min_events=3, min_cluster_span=5))
    assert len(out) == 2


def test_short_cluster_below_min_span_skipped():
    edl = EditDecisionList(decisions=[_d(10, 10.2), _d(10.5, 10.7), _d(11, 11.2)])
    # Three events tightly packed but span < min_cluster_span — should not emit.
    out = find_clusters(edl, DensityParams(window_seconds=60, min_events=3, min_cluster_span=10))
    assert len(out) == 0


def test_categories_aggregated():
    edl = EditDecisionList(decisions=[
        _d(10, 11, "profanity"), _d(15, 16, "drugs"), _d(25, 26, "sex"),
    ])
    out = find_clusters(edl, DensityParams(window_seconds=60, min_events=3, min_cluster_span=5))
    assert len(out) == 1
    assert "profanity" in out.decisions[0].category
    assert "drugs" in out.decisions[0].category
    assert "sex" in out.decisions[0].category
