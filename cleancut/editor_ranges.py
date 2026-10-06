"""Pure range/EDL arithmetic — no subprocess calls, no ffmpeg."""

from __future__ import annotations

import math
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
    cuts = normalize_cuts(cuts, duration)
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
    return [r for r in kept if r.duration > 0]


def normalize_cuts(ranges: list[Range], duration: float | None = None) -> list[Range]:
    """The single cut plan used by rendering and every timestamp mapper.

    Cuts are half-open [start, end), clipped to the source, and unioned. Never
    subtract overlapping or out-of-bounds removed time twice.
    """
    clipped = []
    for r in ranges:
        if not math.isfinite(r.start) or not math.isfinite(r.end):
            raise ValueError("Cut timestamps must be finite")
        start, end = max(0.0, r.start), r.end
        if duration is not None:
            end = min(end, duration)
        if end > start:
            clipped.append(Range(start, end))
    merged: list[Range] = []
    for r in sorted(clipped, key=lambda r: r.start):
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
    for c in normalize_cuts(cuts):
        if c.start > t:
            break
        if c.start <= t < c.end:
            return None
        out -= c.duration
    return max(0.0, out)


def adjust_subtitles_for_cuts(
    subs: list[Subtitle], cuts: list[Range]
) -> list[Subtitle]:
    """Shift / trim / drop subtitles to match a video with the given cuts removed."""
    if not cuts:
        return list(subs)
    cuts = normalize_cuts(cuts)
    out: list[Subtitle] = []
    next_idx = 1
    for s in subs:
        mapped = _map_interval(Range(s.start, s.end), cuts)
        if mapped is None:
            continue
        out.append(Subtitle(index=next_idx, start=mapped.start, end=mapped.end, text=s.text))
        next_idx += 1
    return out


def shift_ranges_after_cuts(ranges: list[Range], cuts: list[Range]) -> list[Range]:
    """Map mute ranges from source timeline to cut-output timeline."""
    cuts = normalize_cuts(cuts)
    out: list[Range] = []
    for r in ranges:
        mapped = _map_interval(r, cuts)
        if mapped is not None:
            out.append(mapped)
    return out


def _map_interval(r: Range, cuts: list[Range]) -> Range | None:
    """Collapse cut portions to their join, preserving all surviving content.

    Unlike a point lookup, endpoints inside cuts must snap to that join, not
    zero. Endpoints in different cuts can still enclose a surviving interval.
    """
    def collapse(t: float) -> float:
        t = max(0.0, t)
        return t - sum(max(0.0, min(t, c.end) - c.start) for c in cuts)

    start, end = collapse(r.start), collapse(r.end)
    return Range(start, end) if end - start > 1e-9 else None


def edl_to_ranges(edl: EditDecisionList, action: str) -> list[Range]:
    return [Range(d.start, d.end) for d in edl.by_action(action)]
