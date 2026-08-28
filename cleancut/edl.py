from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from cleancut.scenes import Shot


@dataclass
class EditDecision:
    start: float            # seconds
    end: float              # seconds
    action: str             # "mute" | "cut" | "keep"
    category: str           # "profanity" | "drugs" | "sex" | "violence" | "nudity"
    reason: str = ""        # short human-readable why
    text_before: str = ""   # original subtitle text (if dialogue-based)
    text_after: str = ""    # softened subtitle text (if dialogue-based)
    source: str = ""        # "subtitle" | "whisper" | "visual"
    accepted: bool = True   # GUI review state

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class EditDecisionList:
    decisions: list[EditDecision] = field(default_factory=list)
    video_path: str = ""
    subtitle_path: str = ""

    def __len__(self) -> int:
        return len(self.decisions)

    def __iter__(self):
        return iter(self.decisions)

    def add(self, d: EditDecision) -> None:
        self.decisions.append(d)

    def extend(self, ds: Iterable[EditDecision]) -> None:
        self.decisions.extend(ds)

    def sorted(self) -> EditDecisionList:
        return EditDecisionList(
            decisions=sorted(self.decisions, key=lambda d: (d.start, d.end)),
            video_path=self.video_path,
            subtitle_path=self.subtitle_path,
        )

    def by_action(self, action: str) -> list[EditDecision]:
        return [d for d in self.decisions if d.action == action and d.accepted]

    def pad(self, seconds: float) -> EditDecisionList:
        out = []
        for d in self.decisions:
            out.append(
                EditDecision(
                    start=max(0.0, d.start - seconds),
                    end=d.end + seconds,
                    action=d.action,
                    category=d.category,
                    reason=d.reason,
                    text_before=d.text_before,
                    text_after=d.text_after,
                    source=d.source,
                    accepted=d.accepted,
                )
            )
        return EditDecisionList(decisions=out, video_path=self.video_path, subtitle_path=self.subtitle_path)

    def merge_overlapping(self, gap: float = 0.0) -> EditDecisionList:
        """Merge adjacent decisions of the same action. 'cut' wins over 'mute'."""
        if not self.decisions:
            return EditDecisionList(video_path=self.video_path, subtitle_path=self.subtitle_path)
        ranked = {"keep": 0, "mute": 1, "cut": 2}
        items = sorted(self.decisions, key=lambda d: d.start)
        # Copy before extending in place — callers' decision objects (e.g. ones
        # loaded from an EDL file) must not be silently modified.
        merged: list[EditDecision] = [replace(items[0])]
        for d in items[1:]:
            last = merged[-1]
            if d.start <= last.end + gap:
                # Overlap or near-touching: merge.
                new_action = last.action if ranked[last.action] >= ranked[d.action] else d.action
                last.end = max(last.end, d.end)
                last.action = new_action
                # Concatenate reasons / categories distinctly.
                if d.category not in last.category:
                    last.category = f"{last.category}+{d.category}"
                if d.reason and d.reason not in last.reason:
                    last.reason = f"{last.reason}; {d.reason}".strip("; ")
                if d.source and d.source not in last.source:
                    last.source = f"{last.source}+{d.source}"
            else:
                merged.append(replace(d))
        return EditDecisionList(
            decisions=merged, video_path=self.video_path, subtitle_path=self.subtitle_path
        )

    def to_json(self, path: Path) -> None:
        payload = {
            "video_path": self.video_path,
            "subtitle_path": self.subtitle_path,
            "decisions": [asdict(d) for d in self.decisions],
        }
        Path(path).write_text(json.dumps(payload, indent=2))

    @classmethod
    def from_json(cls, path: Path) -> EditDecisionList:
        data = json.loads(Path(path).read_text())
        return cls(
            decisions=[EditDecision(**d) for d in data.get("decisions", [])],
            video_path=data.get("video_path", ""),
            subtitle_path=data.get("subtitle_path", ""),
        )

    def summary(self) -> dict[str, int]:
        """Counts accepted decisions only — must agree with what renders."""
        out: dict[str, int] = {}
        for d in self.decisions:
            if not d.accepted:
                continue
            key = f"{d.action}:{d.category.split('+')[0]}"
            out[key] = out.get(key, 0) + 1
        return out


def snap_edl_to_shots(edl: EditDecisionList, shots: list["Shot"]) -> EditDecisionList:
    """Extend each `cut` decision outward to enclosing shot boundaries.

    Mutes are left alone — they're audio-only and should be word-precise.
    """
    if not shots:
        return edl
    from cleancut.scenes import snap_range_to_shots

    out: list[EditDecision] = []
    for d in edl.decisions:
        if d.action == "cut":
            ns, ne = snap_range_to_shots(d.start, d.end, shots)
            out.append(
                EditDecision(
                    start=ns,
                    end=ne,
                    action=d.action,
                    category=d.category,
                    reason=(d.reason + " | snapped-to-shot").strip(" |"),
                    text_before=d.text_before,
                    text_after=d.text_after,
                    source=d.source,
                    accepted=d.accepted,
                )
            )
        else:
            out.append(d)
    return EditDecisionList(decisions=out, video_path=edl.video_path, subtitle_path=edl.subtitle_path)
