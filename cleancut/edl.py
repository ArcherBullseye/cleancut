from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from cleancut.scenes import Shot


@dataclass
class EditDecision:
    start: float            # seconds
    end: float              # seconds
    action: str             # "mute" | "replace" | "cut" | "keep"
    category: str           # "profanity" | "drugs" | "sex" | "violence" | "nudity"
    reason: str = ""        # short human-readable why
    text_before: str = ""   # original subtitle text (if dialogue-based)
    text_after: str = ""    # softened subtitle text (if dialogue-based)
    source: str = ""        # "subtitle" | "whisper" | "visual"
    accepted: bool = True   # GUI review state
    # Unpadded word intervals survive merging, padding, and browser review.
    # Old EDLs have none and safely remain mute-only until scanned again.
    word_edits: list[dict] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


# Detectors label a hit with a composite category ("violence+profanity"), while
# the user configures an action per base category. The strongest action any
# component asks for wins, so a scene that is both keep-violence and cut-sex is
# still cut, and one that is only keep-violence is dropped.
_ACTION_STRENGTH = {"keep": 0, "replace": 1, "mute": 2, "cut": 3}


def resolve_action(
    category: str,
    actions: dict[str, str] | None,
    default: str = "cut",
) -> str:
    """Resolve a detector's category against the configured per-category actions.

    Returns `default` when `actions` is empty or nothing in `category` maps to a
    configured entry -- the LLM's catch-all "multi" has no setting of its own.
    Callers MUST drop the decision when this returns "keep": every detector used
    to hardcode action="cut", which silently overrode the user's settings.
    """
    if not actions:
        return default
    resolved = [actions[part] for part in category.split("+") if part in actions]
    if not resolved:
        return default
    return max(resolved, key=lambda a: _ACTION_STRENGTH.get(a, 0))


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

    def audio_edits(self) -> list[EditDecision]:
        """Both actions remove original speech; replacement overlays are optional."""
        return [d for d in self.decisions if d.action in {"mute", "replace"} and d.accepted]

    def pad(self, seconds: float, *, word_end_padding_ms: int = 200) -> EditDecisionList:
        out = []
        for d in self.decisions:
            word_timed = d.source == "whisper-word" and bool(d.word_edits)
            padding = min(seconds, 0.04) if word_timed else seconds
            end = d.end + padding
            if word_timed:
                # Keep unpadded word timings for synthesis; only censorship
                # receives this guard. Anchor it to the last word, not an
                # already padded decision, so repeated padding cannot grow it.
                last_word = max(d.word_edits, key=lambda w: float(w["end"]))
                word_end = float(last_word["end"])
                end = word_end + max(0, min(500, word_end_padding_ms)) / 1000
                following = last_word.get("next_word_start")
                if following is not None and math.isfinite(float(following)):
                    end = min(end, max(word_end, float(following)))
                end = max(d.end, end)
            out.append(
                EditDecision(
                    start=max(0.0, d.start - padding),
                    end=end,
                    action=d.action,
                    category=d.category,
                    reason=d.reason,
                    text_before=d.text_before,
                    text_after=d.text_after,
                    source=d.source,
                    accepted=d.accepted,
                    word_edits=d.word_edits,
                )
            )
        return EditDecisionList(decisions=out, video_path=self.video_path, subtitle_path=self.subtitle_path)

    def merge_overlapping(self, gap: float = 0.0) -> EditDecisionList:
        """Merge adjacent decisions of the same action. 'cut' wins over 'mute'."""
        if not self.decisions:
            return EditDecisionList(video_path=self.video_path, subtitle_path=self.subtitle_path)
        ranked = _ACTION_STRENGTH
        # Review-only suggestions must never enlarge or change an accepted
        # edit. Merge each acceptance state independently, then restore time order.
        items = sorted(self.decisions, key=lambda d: (not d.accepted, d.start))
        # Copy before extending in place — callers' decision objects (e.g. ones
        # loaded from an EDL file) must not be silently modified.
        merged: list[EditDecision] = [replace(items[0])]
        for d in items[1:]:
            last = merged[-1]
            word_mutes = (d.action in {"mute", "replace"} and last.action in {"mute", "replace"}
                          and d.source == last.source == "whisper-word"
                          and d.word_edits and last.word_edits)
            # Never merge adjacent, differently chosen audio actions. A true
            # overlap still resolves conservatively (mute wins over replace).
            different_audio = {d.action, last.action} == {"mute", "replace"}
            merge_gap = 0.0 if word_mutes or different_audio else gap
            overlaps = d.start < last.end if different_audio else d.start <= last.end + merge_gap
            if overlaps and d.accepted == last.accepted:
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
                last.word_edits = [*last.word_edits, *d.word_edits]
            else:
                merged.append(replace(d))
        return EditDecisionList(
            decisions=sorted(merged, key=lambda d: (d.start, d.end)),
            video_path=self.video_path, subtitle_path=self.subtitle_path
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
    """Snap broad accepted scene cuts, retaining temporal/word precision.

    NudeNet ranges, word edits, and unselected suggestions are left alone.
    """
    if not shots:
        return edl
    from cleancut.scenes import snap_range_to_shots

    out: list[EditDecision] = []
    for d in edl.decisions:
        # NudeNet already supplies temporal margins; snapping can turn a
        # brief mistaken detection into an entire missing shot. Word cuts
        # likewise need to retain word precision.
        precise = bool(set(d.source.split("+")) & {"visual", "visual-shot", "visual-temporal", "whisper-word"})
        if d.action == "cut" and d.accepted and not precise:
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
                    word_edits=d.word_edits,
                )
            )
        else:
            out.append(d)
    return EditDecisionList(decisions=out, video_path=edl.video_path, subtitle_path=edl.subtitle_path)
