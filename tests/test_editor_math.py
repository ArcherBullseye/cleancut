from cleancut.editor import Range, adjust_subtitles_for_cuts, keep_segments, shift_after_cuts, shift_ranges_after_cuts
from cleancut.subtitles import Subtitle


def test_keep_segments_basic():
    segs = keep_segments(100.0, [Range(10, 20), Range(40, 50)])
    assert [(s.start, s.end) for s in segs] == [(0.0, 10.0), (20.0, 40.0), (50.0, 100.0)]


def test_keep_segments_at_start():
    segs = keep_segments(100.0, [Range(0, 10)])
    assert [(s.start, s.end) for s in segs] == [(10.0, 100.0)]


def test_keep_segments_at_end():
    segs = keep_segments(100.0, [Range(90, 100)])
    assert [(s.start, s.end) for s in segs] == [(0.0, 90.0)]


def test_shift_after_cuts():
    cuts = [Range(10, 20), Range(40, 50)]
    assert shift_after_cuts(5, cuts) == 5
    assert shift_after_cuts(25, cuts) == 15        # 25 - 10s removed
    assert shift_after_cuts(60, cuts) == 40        # 60 - 20s removed
    assert shift_after_cuts(15, cuts) is None      # inside a cut


def test_adjust_subtitles_for_cuts_drops_and_shifts():
    cuts = [Range(10, 20)]
    subs = [
        Subtitle(index=1, start=5, end=7, text="before"),
        Subtitle(index=2, start=12, end=15, text="inside cut"),
        Subtitle(index=3, start=25, end=27, text="after"),
    ]
    adjusted = adjust_subtitles_for_cuts(subs, cuts)
    assert len(adjusted) == 2
    assert adjusted[0].text == "before"
    assert adjusted[1].text == "after"
    assert adjusted[1].start == 15
    assert adjusted[1].end == 17


def test_shift_ranges_after_cuts():
    cuts = [Range(10, 20)]
    mutes = [Range(5, 8), Range(25, 30)]
    shifted = shift_ranges_after_cuts(mutes, cuts)
    assert [(r.start, r.end) for r in shifted] == [(5, 8), (15, 20)]
