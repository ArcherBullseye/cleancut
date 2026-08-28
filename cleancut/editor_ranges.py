"""Pure range/EDL arithmetic — no subprocess calls, no ffmpeg."""

from __future__ import annotations

from dataclasses import dataclass

from cleancut.edl import EditDecisionList
from cleancut.subtitles import Subtitle


@dataclass
class Range:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def keep_segments(duration: float, cuts: list[Range]) -> list[Range]:
    """Complement of cuts within [0, duration]. Returns the segments we keep."""
    cuts = sorted(cuts, key=lambda r: r.start)
    kept: list[Range] = []
    cursor = 0.0
    for c in cuts:
        s = max(c.start, 0.0)
        e = min(c.end, duration)
        if e <= cursor:
            continue
        if s > cursor:
            kept.append(Range(cursor, s))
        cursor = max(cursor, e)
    if cursor < duration:
        kept.append(Range(cursor, duration))
    return [r for r in kept if r.duration > 0.001]


def _merge_ranges(ranges: list[Range]) -> list[Range]:
    """Sort and union overlapping/touching ranges."""
    merged: list[Range] = []
    for r in sorted(ranges, key=lambda r: r.start):
        if merged and r.start <= merged[-1].end:
            merged[-1] = Range(merged[-1].start, max(merged[-1].end, r.end))
        else:
            merged.append(Range(r.start, r.end))
    return merged


def shift_after_cuts(t: float, cuts: list[Range]) -> float | None:
    """Map a source-timeline timestamp to the cut-output timeline.

    Returns None if `t` falls inside a removed segment. Overlapping cuts are
    unioned first — subtracting each duration would double-count the overlap.
    """
    out = t
    for c in _merge_ranges(cuts):
        if c.start > t:
            break
        if c.start <= t <= c.end:
            return None
        out -= c.duration
    return max(0.0, out)


def adjust_subtitles_for_cuts(
    subs: list[Subtitle], cuts: list[Range]
) -> list[Subtitle]:
    """Shift / trim / drop subtitles to match a video with the given cuts removed."""
    if not cuts:
        return list(subs)
    cuts = sorted(cuts, key=lambda r: r.start)
    out: list[Subtitle] = []
    next_idx = 1
    for s in subs:
        new_start = shift_after_cuts(s.start, cuts)
        new_end = shift_after_cuts(s.end, cuts)
        # Both endpoints fall inside cuts -> drop.
        if new_start is None and new_end is None:
            continue
        # Start cut out: snap to the next keep boundary.
        if new_start is None:
            for c in cuts:
                if c.start <= s.start <= c.end:
                    snapped = shift_after_cuts(c.end + 1e-4, cuts) or 0.0
                    new_start = snapped
                    break
        if new_end is None:
            for c in cuts:
                if c.start <= s.end <= c.end:
                    snapped = shift_after_cuts(c.start - 1e-4, cuts)
                    new_end = snapped if snapped is not None else new_start
                    break
        if new_start is None or new_end is None or new_end <= new_start:
            continue
        out.append(Subtitle(index=next_idx, start=new_start, end=new_end, text=s.text))
        next_idx += 1
    return out


def shift_ranges_after_cuts(ranges: list[Range], cuts: list[Range]) -> list[Range]:
    """Map mute ranges from source timeline to cut-output timeline."""
    out: list[Range] = []
    for r in ranges:
        ns = shift_after_cuts(r.start, cuts)
        ne = shift_after_cuts(r.end, cuts)
        if ns is None and ne is None:
            continue
        if ns is None:
            ns = 0.0
        if ne is None:
            # Range tail falls inside a cut: trim to the cut boundary.
            for c in sorted(cuts, key=lambda x: x.start):
                if c.start <= r.end <= c.end:
                    snapped = shift_after_cuts(c.start - 1e-4, cuts)
                    if snapped is not None:
                        ne = snapped
                    break
        if ne is None or ne <= ns:
            continue
        out.append(Range(ns, ne))
    return out


def edl_to_ranges(edl: EditDecisionList, action: str) -> list[Range]:
    return [Range(d.start, d.end) for d in edl.by_action(action)]
