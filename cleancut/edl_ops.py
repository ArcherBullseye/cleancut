"""Helpers for hand-editing an EDL (used by `add-cut` and `review` subcommands)."""

from __future__ import annotations

import re

from cleancut.constants import FOCAL_CATEGORIES
from cleancut.edl import EditDecision, EditDecisionList


def parse_timestamp(s: str) -> float:
    """Parse "MM:SS", "MM:SS.mmm", "H:MM:SS", "H:MM:SS.mmm", or plain seconds."""
    s = s.strip()
    # Plain seconds
    if re.fullmatch(r"\d+(\.\d+)?", s):
        return float(s)
    parts = s.split(":")
    if len(parts) == 2:
        m, sec = parts
        return int(m) * 60 + float(sec)
    if len(parts) == 3:
        h, m, sec = parts
        return int(h) * 3600 + int(m) * 60 + float(sec)
    raise ValueError(f"Cannot parse timestamp {s!r}; use SS, MM:SS, or H:MM:SS")


def fmt_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds - h * 3600 - m * 60
    return f"{h}:{m:02d}:{s:05.2f}" if h else f"{m}:{s:05.2f}"


def fmt_ffmpeg_timestamp(seconds: float) -> str:
    """HH:MM:SS.mmm — the zero-padded form ffmpeg's -ss expects."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds - h * 3600 - m * 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def cuts_for_review(
    edl: EditDecisionList, include_violence: bool = False,
) -> list[EditDecision]:
    """Accepted cut decisions in start order; violence-only cuts hidden unless asked."""
    cuts = [d for d in edl.decisions if d.action == "cut" and d.accepted]
    if not include_violence:
        cuts = [d for d in cuts if set(d.category.split("+")) & FOCAL_CATEGORIES]
    return sorted(cuts, key=lambda d: d.start)
