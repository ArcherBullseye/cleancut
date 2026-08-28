"""Density-based clustering of EDL events.

A scene is rarely a single offensive word — it's a *cluster* of them. When N or
more wordlist hits fall within a rolling W-second window, we infer there's a
high-content scene there and emit a single `cut` covering the cluster. Cheap,
no model required, catches "the dealer scene" class of content where context
is dense but each individual hit is borderline.
"""

from __future__ import annotations

from dataclasses import dataclass

from cleancut.edl import EditDecision, EditDecisionList


# Density is a *dialogue density* signal. Visual hits (NudeNet, VLM) are excluded
# from clustering because they shouldn't pull unrelated dialogue mutes into a
# single oversized cut (see v5 cut #2 ballooning to 65s from a single NudeNet FP).
DIALOGUE_SOURCES = {
    "subtitle",
    "whisper-word",
    "llm-dialogue",
    "audio",        # audio events are content-bearing too
}


@dataclass
class DensityParams:
    window_seconds: float = 60.0   # rolling window
    min_events: int = 3            # min hits in window to count as a cluster
    pad_seconds: float = 1.0       # extend cluster by this on each side
    # If the cluster spans more than this, treat it as a "scene" worth cutting.
    min_cluster_span: float = 8.0


def _is_dialogue_event(d) -> bool:
    """True if the event's source qualifies it for density clustering."""
    # Source may be a combined string from prior merges ("subtitle+vlm").
    return any(tok in DIALOGUE_SOURCES for tok in d.source.split("+"))


def find_clusters(edl: EditDecisionList, params: DensityParams) -> EditDecisionList:
    """Find dense clusters in `edl` and emit one `cut` per cluster.

    Only considers `mute`/`cut` decisions whose source is a dialogue or audio
    signal — never visual. This prevents a single NudeNet/VLM hit from
    anchoring a cluster and pulling surrounding wordlist mutes into a giant cut.
    """
    events = sorted(
        [d for d in edl.decisions
         if d.action in ("mute", "cut") and d.accepted and _is_dialogue_event(d)],
        key=lambda d: d.start,
    )
    if len(events) < params.min_events:
        return EditDecisionList(video_path=edl.video_path, subtitle_path=edl.subtitle_path)

    out: list[EditDecision] = []
    n = len(events)
    i = 0
    while i < n:
        # Find the furthest event that starts within window_seconds of events[i].
        window_end = events[i].start + params.window_seconds
        j = i
        while j + 1 < n and events[j + 1].start <= window_end:
            j += 1
        # Number of events in this window.
        in_window = j - i + 1
        if in_window >= params.min_events:
            # Extend the cluster as long as each next event still keeps density.
            cluster_start = events[i].start
            # Events are sorted by start — an early long event can end last.
            cluster_end = max(e.end for e in events[i:j + 1])
            k = j
            while k + 1 < n:
                # Check if adding events[k+1] keeps a rolling density.
                rolling_start = max(cluster_start, events[k + 1].start - params.window_seconds)
                count = sum(1 for e in events[i:k + 2] if e.start >= rolling_start)
                if count >= params.min_events:
                    k += 1
                    cluster_end = max(cluster_end, events[k].end)
                else:
                    break
            span = cluster_end - cluster_start
            if span >= params.min_cluster_span:
                cats = sorted({e.category.split("+")[0] for e in events[i:k + 1]})
                out.append(
                    EditDecision(
                        start=max(0.0, cluster_start - params.pad_seconds),
                        end=cluster_end + params.pad_seconds,
                        action="cut",
                        category="+".join(cats),
                        reason=f"density: {k - i + 1} events in {span:.1f}s window",
                        source="density",
                    )
                )
            i = k + 1
        else:
            i += 1

    return EditDecisionList(decisions=out, video_path=edl.video_path, subtitle_path=edl.subtitle_path)
