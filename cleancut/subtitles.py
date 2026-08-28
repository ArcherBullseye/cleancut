from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import srt

from cleancut.config import Config
from cleancut.edl import EditDecision, EditDecisionList


@dataclass
class Subtitle:
    index: int
    start: float            # seconds
    end: float              # seconds
    text: str

    @classmethod
    def from_srt(cls, s: srt.Subtitle) -> Subtitle:
        return cls(
            index=s.index,
            start=s.start.total_seconds(),
            end=s.end.total_seconds(),
            text=s.content,
        )


def read_srt(path: Path) -> list[Subtitle]:
    raw = Path(path).read_text(encoding="utf-8", errors="replace")
    return [Subtitle.from_srt(s) for s in srt.parse(raw)]


def write_srt(subs: list[Subtitle], path: Path) -> None:
    import datetime as dt

    items = [
        srt.Subtitle(
            index=s.index,
            start=dt.timedelta(seconds=s.start),
            end=dt.timedelta(seconds=s.end),
            content=s.text,
        )
        for s in subs
    ]
    Path(path).write_text(srt.compose(items), encoding="utf-8")


def _compile_patterns(
    wordlists: dict[str, list],
) -> dict[str, list[tuple[re.Pattern, str]]]:
    """Compile patterns; each entry returned as (pattern, strength).

    Accepts both legacy flat-string form and dict form with `strength`.
    """
    out: dict[str, list[tuple[re.Pattern, str]]] = {}
    for cat, entries in wordlists.items():
        compiled: list[tuple[re.Pattern, str]] = []
        for entry in entries:
            if isinstance(entry, str):
                compiled.append((re.compile(entry, re.IGNORECASE), "strong"))
            else:
                pat = entry["pattern"]
                strength = entry.get("strength", "strong")
                compiled.append((re.compile(pat, re.IGNORECASE), strength))
        out[cat] = compiled
    return out


# Context-gating window: a weak hit only counts if a strong hit (any category) lies
# within this many seconds on either side. 30s ≈ one dialogue exchange.
CONTEXT_WINDOW_SECONDS = 30.0

# When several categories match the same line/window, the highest wins.
CATEGORY_SEVERITY = {"profanity": 1, "violence": 2, "drugs": 3, "sex": 4, "nudity": 5}


def _filter_weak_without_context(
    edl: EditDecisionList, window: float = CONTEXT_WINDOW_SECONDS,
) -> EditDecisionList:
    """Drop any decision whose reason starts with 'weak:' and which has no
    strong neighbor within `window` seconds on either side."""
    if not edl.decisions:
        return edl
    strong_times = sorted(
        (d.start, d.end) for d in edl.decisions
        if not d.reason.startswith("weak:")
    )
    if not strong_times:
        # Nothing strong — drop all weak hits.
        kept = [d for d in edl.decisions if not d.reason.startswith("weak:")]
        return EditDecisionList(
            decisions=kept, video_path=edl.video_path, subtitle_path=edl.subtitle_path,
        )
    kept = []
    for d in edl.decisions:
        if not d.reason.startswith("weak:"):
            kept.append(d)
            continue
        # Weak hit — check for any strong hit within window.
        has_context = any(
            s_end + window >= d.start and s_start - window <= d.end
            for s_start, s_end in strong_times
        )
        if has_context:
            kept.append(d)
    return EditDecisionList(
        decisions=kept, video_path=edl.video_path, subtitle_path=edl.subtitle_path,
    )


def scan_subtitles(subs: list[Subtitle], config: Config) -> EditDecisionList:
    """Match wordlists against subtitle text. Each match -> one EditDecision.

    Weak (ambiguous) patterns only survive if a strong hit lies within
    CONTEXT_WINDOW_SECONDS on either side.
    """
    patterns = _compile_patterns(config.wordlists)
    edl = EditDecisionList()

    for sub in subs:
        matches: list[tuple[str, str, str]] = []  # (category, matched_text, strength)
        for category, pats in patterns.items():
            if category not in config.enabled_categories:
                continue
            for pat, strength in pats:
                for m in pat.finditer(sub.text):
                    matches.append((category, m.group(0), strength))

        if not matches:
            continue

        # Prefer strong matches; break ties by severity.
        best = sorted(
            matches, key=lambda x: (x[2] != "strong", -CATEGORY_SEVERITY.get(x[0], 0))
        )[0]
        category, _, line_strength = best
        action = config.actions.get(category, "mute")
        if action == "keep":
            continue

        text_after = soften_text(sub.text, config.replacements)
        reason_prefix = "weak: " if line_strength == "weak" else "matched: "
        edl.add(
            EditDecision(
                start=sub.start,
                end=sub.end,
                action=action,
                category=category,
                reason=reason_prefix + ", ".join(sorted(set(m[1].lower() for m in matches))),
                text_before=sub.text,
                text_after=text_after,
                source="subtitle",
            )
        )

    return _filter_weak_without_context(edl)


def soften_text(text: str, replacements: dict[str, str]) -> str:
    """Case-preserving word-boundary substitution, in a single pass.

    A single combined regex guarantees replacements never re-substitute each
    other's output (shit→crap plus crap→crud must yield "crap", not "crud").
    """
    if not replacements:
        return text
    # Longer phrases first so multi-word entries win the alternation.
    keys = sorted(replacements.keys(), key=len, reverse=True)
    pattern = re.compile(
        r"\b(?:" + "|".join(re.escape(k) for k in keys) + r")\b", re.IGNORECASE
    )
    lower_map = {k.lower(): v for k, v in replacements.items()}

    def _sub(m: re.Match) -> str:
        original = m.group(0)
        return _match_case(original, lower_map[original.lower()])

    return pattern.sub(_sub, text)


def _match_case(original: str, replacement: str) -> str:
    if original.isupper():
        return replacement.upper()
    if original[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def softened_subtitles(subs: list[Subtitle], config: Config) -> list[Subtitle]:
    """Return a copy of subs with every line softened by the replacements map."""
    return [
        Subtitle(index=s.index, start=s.start, end=s.end, text=soften_text(s.text, config.replacements))
        for s in subs
    ]


def scan_words(words, config: Config) -> EditDecisionList:
    """Match wordlists against individual word-level Whisper output.

    Produces tight mute ranges around just the offending word(s), instead of
    the entire subtitle line. Adjacent flagged words within the same phrase
    will be merged later in the pipeline.

    `words` is a list of cleancut.transcribe.Word; imported lazily to avoid
    a hard dep on the whisper extras.
    """
    patterns = _compile_patterns(config.wordlists)
    edl = EditDecisionList()

    n = len(words)
    for i, w in enumerate(words):
        for window_size in (1, 2, 3):
            j = i + window_size
            if j > n:
                break
            window = words[i:j]
            text = " ".join(x.text for x in window).strip(" .,!?-")
            if not text:
                continue
            best: tuple[str, str, str] | None = None  # (category, matched, strength)
            best_score = (-1, False)  # (severity, is_strong)
            for category, pats in patterns.items():
                if category not in config.enabled_categories:
                    continue
                for pat, strength in pats:
                    m = pat.search(text)
                    if m:
                        sev = CATEGORY_SEVERITY.get(category, 0)
                        score = (sev, strength == "strong")
                        if score > best_score:
                            best = (category, m.group(0), strength)
                            best_score = score
            if best is None:
                continue
            category, matched, strength = best
            action = config.actions.get(category, "mute")
            if action == "keep":
                continue
            reason_prefix = "weak: " if strength == "weak" else "matched: "
            edl.add(
                EditDecision(
                    start=window[0].start,
                    end=window[-1].end,
                    action=action,
                    category=category,
                    reason=reason_prefix + matched.lower(),
                    text_before=text,
                    text_after=soften_text(text, config.replacements),
                    source="whisper-word",
                )
            )
            break

    return _filter_weak_without_context(edl)
