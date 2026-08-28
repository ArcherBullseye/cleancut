from cleancut.config import Config
from cleancut.subtitles import Subtitle, scan_subtitles


def _sub(idx, start, end, text):
    return Subtitle(index=idx, start=start, end=end, text=text)


def test_lone_weak_hit_dropped():
    config = Config.load_defaults()
    subs = [
        _sub(1, 0, 2, "Take a deep blow to the head"),
        _sub(2, 60, 62, "It is what it is"),
    ]
    edl = scan_subtitles(subs, config)
    assert len(edl) == 0


def test_weak_hit_kept_when_strong_neighbor_present():
    config = Config.load_defaults()
    subs = [
        _sub(1, 0, 2, "He sells cocaine"),       # STRONG drug
        _sub(2, 10, 12, "Snort the blow"),       # WEAK drug — within 30s of strong
    ]
    edl = scan_subtitles(subs, config)
    cats = [d.category for d in edl]
    assert "drugs" in cats
    # Both lines should be flagged.
    assert len(edl) == 2


def test_strong_hits_always_kept():
    config = Config.load_defaults()
    subs = [
        _sub(1, 0, 2, "Cocaine deal at midnight"),
        _sub(2, 600, 602, "Heroin overdose victim"),  # both strong, 10 min apart, both kept
    ]
    edl = scan_subtitles(subs, config)
    assert len(edl) == 2


def test_weak_neighbors_each_other_dropped():
    config = Config.load_defaults()
    # Two weak hits near each other but no strong context — both dropped.
    subs = [
        _sub(1, 0, 2, "Pass the joint"),
        _sub(2, 5, 7, "Smoke the bong"),
    ]
    edl = scan_subtitles(subs, config)
    assert len(edl) == 0


def test_profanity_unaffected_by_gating():
    config = Config.load_defaults()
    subs = [_sub(1, 0, 2, "What the fuck")]
    edl = scan_subtitles(subs, config)
    assert len(edl) == 1
    assert edl.decisions[0].category == "profanity"
