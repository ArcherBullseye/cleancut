"""Cross-signal corroboration for low-trust detectors.

A solo VLM flag (a single-frame visual judgment about an unconfirmed scene)
should not become a cut without supporting evidence from another detector.
This module marks solo-VLM and solo-NudeNet decisions as `accepted=false`
unless another signal lies within `radius_seconds` on either side.

The user can keep them via the interactive review (they remain in the EDL,
just hidden), or globally with --allow-solo-visual.
"""

from __future__ import annotations

from cleancut.edl import EditDecisionList

# Source tokens that, when present alongside a visual source, count as corroboration.
CORROBORATING_SOURCES = {
    "subtitle",
    "whisper-word",
    "llm-dialogue",
    "audio",
    "density",
    "manual",
}

VISUAL_ONLY_SOURCES = {"vlm", "visual", "visual-shot"}


def _is_visual_only(source: str) -> bool:
    """Source string contains only visual signals (no dialogue/audio token)."""
    tokens = set(source.split("+"))
    if not tokens & VISUAL_ONLY_SOURCES:
        return False
    return not (tokens & CORROBORATING_SOURCES)


def mark_unsupported_visual(
    edl: EditDecisionList,
    radius_seconds: float = 5.0,
) -> tuple[EditDecisionList, int]:
    """Set accepted=false on visual-only decisions with no corroborating neighbor.

    Returns (updated_edl, n_marked). Corroborators must already be accepted
    themselves to count.
    """
    if not edl.decisions:
        return edl, 0

    corrob_times: list[tuple[float, float]] = sorted(
        (d.start, d.end) for d in edl.decisions
        if d.accepted and any(tok in CORROBORATING_SOURCES for tok in d.source.split("+"))
    )

    n_marked = 0
    for d in edl.decisions:
        if not d.accepted:
            continue
        if not _is_visual_only(d.source):
            continue
        # Check for any corroborating event within ±radius of this cut.
        has_support = any(
            cs - radius_seconds <= d.end and ce + radius_seconds >= d.start
            for cs, ce in corrob_times
        )
        if not has_support:
            d.accepted = False
            d.reason = f"[solo-visual; needs corroboration] {d.reason}"[:200]
            n_marked += 1
    return edl, n_marked
