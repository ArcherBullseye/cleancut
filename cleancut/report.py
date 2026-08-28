"""Human-readable reports for `inspect` (plan) and `scan`/`clean` (results)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from cleancut.config import Config
from cleancut.constants import MAX_REASON_LENGTH
from cleancut.edl import EditDecisionList
from cleancut.edl_ops import fmt_timestamp
from cleancut.probe import (
    IMAGE_SUBTITLE_CODECS,
    TEXT_SUBTITLE_CODECS,
    Stream,
    find_sidecar_subtitle,
    pick_audio_track,
    pick_embedded_subtitle,
    subtitle_streams,
)


def _fmt_time(seconds: float) -> str:
    return fmt_timestamp(max(0.0, seconds))


def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m = int(seconds // 60)
    s = seconds - m * 60
    if m < 60:
        return f"{m}m {s:.1f}s"
    h = m // 60
    m = m % 60
    return f"{h}h {m}m {s:.1f}s"


def build_plan(
    video: Path,
    streams: list[Stream],
    config: Config,
    *,
    audio_track: int | None,
    prefer_language: str,
    use_visual: bool,
    use_scenes: bool,
    use_whisper: bool,
    burn_subs: bool,
    explicit_subs: Path | None = None,
) -> str:
    """Render a plain-text 'what will be changed' plan for the `inspect` command."""
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("CLEANCUT PLAN")
    lines.append("=" * 70)
    lines.append(f"Source: {video}")
    lines.append("")

    # Subtitle source resolution
    lines.append("Subtitle source")
    lines.append("-" * 70)
    if explicit_subs:
        lines.append(f"  Explicit --subs: {explicit_subs}")
    else:
        sidecar = find_sidecar_subtitle(video, prefer_language=prefer_language)
        if sidecar:
            lines.append(f"  Sidecar .srt:    {sidecar}")
        else:
            lines.append("  No sidecar .srt found.")
            embedded = pick_embedded_subtitle(streams, prefer_language=prefer_language)
            if embedded:
                lines.append(
                    f"  Embedded sub:    stream {embedded.index} "
                    f"({embedded.codec_name}, {embedded.language}) → extract via ffmpeg"
                )
            else:
                text_subs = [s for s in subtitle_streams(streams)
                             if s.codec_name in TEXT_SUBTITLE_CODECS]
                image_subs = [s for s in subtitle_streams(streams)
                              if s.codec_name in IMAGE_SUBTITLE_CODECS]
                if image_subs and not text_subs:
                    lines.append(
                        f"  {len(image_subs)} image-based subtitle track(s) "
                        f"({image_subs[0].codec_name}) — cleancut skips these (no OCR)."
                    )
                else:
                    lines.append("  No usable embedded text subtitles.")
                if use_whisper:
                    try:
                        track = pick_audio_track(streams, audio_track, prefer_language=prefer_language)
                    except ValueError as e:
                        lines.append(f"  Whisper:         ERROR {e}")
                        track = None
                    if track:
                        device = _resolve_planned_device(config)
                        lines.append(
                            f"  Whisper:         audio stream {track.index} "
                            f"({track.codec_name}, {track.language}) "
                            f"→ model={config.whisper_model} device={device} "
                            f"word_ts={config.whisper_word_timestamps}"
                        )
                else:
                    lines.append("  Whisper:         disabled (--no-whisper)")
    lines.append("")

    # Categories and actions
    lines.append("What gets touched")
    lines.append("-" * 70)
    for cat in ["profanity", "drugs", "sex", "violence", "nudity"]:
        action = config.actions.get(cat, "keep")
        if cat in config.enabled_categories and action != "keep":
            n_patterns = len(config.wordlists.get(cat, [])) if cat != "nudity" else "—"
            lines.append(f"  {cat:10s} → {action:5s}  ({n_patterns} patterns)" if cat != "nudity"
                         else f"  {cat:10s} → {action:5s}  (visual NudeNet)")
        else:
            lines.append(f"  {cat:10s} → keep   (disabled)")
    lines.append("")

    # Detectors
    lines.append("Detectors")
    lines.append("-" * 70)
    lines.append(
        f"  Shot boundaries: {'on' if use_scenes else 'off'} "
        f"(threshold {config.scene_threshold})"
    )
    lines.append(
        f"  Visual NudeNet:  {'on' if use_visual else 'off'} "
        f"(sample {config.visual_sample_seconds}s, threshold {config.visual_threshold}, "
        f"shot-aware fraction {config.visual_shot_hit_fraction})"
    )
    lines.append(
        f"  Snap cuts to shots: {'yes' if config.snap_cuts_to_scenes else 'no'}"
    )
    lines.append("")

    # Render plan
    lines.append("Render plan")
    lines.append("-" * 70)
    lines.append(f"  Encoder:         {config.resolved_encoder()} (q={config.quality})")
    lines.append(f"  Burn-in subs:    {'yes (softened)' if burn_subs else 'no'}")
    lines.append("")
    lines.append("Run with `cleancut clean` (same flags) to actually do it.")
    return "\n".join(lines)


def _resolve_planned_device(config: Config) -> str:
    try:
        from cleancut.transcribe import _resolve_device
        return _resolve_device(config.whisper_device, config.whisper_word_timestamps)
    except Exception:
        return config.whisper_device or "auto"


def build_results_report(
    video: Path,
    output: Path | None,
    edl: EditDecisionList,
    *,
    original_duration: float | None = None,
) -> str:
    """Render a plain-text report after a scan or clean run.

    Lists every decision, totals muted/cut time, and a category breakdown.
    """
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("CLEANCUT REPORT")
    lines.append("=" * 70)
    lines.append(f"Source: {video}")
    if output:
        lines.append(f"Output: {output}")
    lines.append("")

    decisions = [d for d in edl.decisions if d.accepted]
    if not decisions:
        lines.append("No decisions emitted — nothing flagged.")
        return "\n".join(lines)

    n_mute = sum(1 for d in decisions if d.action == "mute")
    n_cut = sum(1 for d in decisions if d.action == "cut")
    t_mute = sum(d.duration for d in decisions if d.action == "mute")
    t_cut = sum(d.duration for d in decisions if d.action == "cut")

    lines.append("Summary")
    lines.append("-" * 70)
    lines.append(f"  Total decisions: {len(decisions)}")
    lines.append(f"  Mutes:           {n_mute}  ({_fmt_duration(t_mute)} of audio)")
    lines.append(f"  Cuts:            {n_cut}  ({_fmt_duration(t_cut)} of video)")
    if original_duration:
        new_dur = max(0.0, original_duration - t_cut)
        pct = (t_cut / original_duration * 100) if original_duration > 0 else 0.0
        lines.append(
            f"  Original length: {_fmt_duration(original_duration)}  "
            f"→ after cuts: {_fmt_duration(new_dur)}  ({pct:.1f}% removed)"
        )
    lines.append("")

    # Per-category breakdown
    by_cat: Counter[str] = Counter()
    by_cat_time: dict[str, float] = {}
    for d in decisions:
        cat = d.category.split("+")[0]
        by_cat[cat] += 1
        by_cat_time[cat] = by_cat_time.get(cat, 0) + d.duration
    lines.append("By category")
    lines.append("-" * 70)
    for cat, n in by_cat.most_common():
        lines.append(f"  {cat:10s}  {n:4d}  ({_fmt_duration(by_cat_time[cat])})")
    lines.append("")

    # Top matched words / phrases
    phrases: Counter[str] = Counter()
    for d in decisions:
        if d.text_before:
            # Just the matched part from `reason: "matched: foo"` if present
            if d.reason.startswith("matched:"):
                for tok in d.reason.removeprefix("matched:").split(","):
                    tok = tok.strip().lower()
                    if tok:
                        phrases[tok] += 1
    if phrases:
        lines.append("Top flagged phrases")
        lines.append("-" * 70)
        for phrase, n in phrases.most_common(15):
            lines.append(f"  {n:4d}  {phrase}")
        lines.append("")

    # Full decision list
    lines.append("Every decision")
    lines.append("-" * 70)
    for i, d in enumerate(decisions, start=1):
        cat = d.category.split("+")[0]
        head = f"[{i:3d}] {_fmt_time(d.start)} – {_fmt_time(d.end)}  {d.action.upper():4s}  {cat:9s}"
        lines.append(head)
        if d.text_before:
            before = d.text_before.replace("\n", " ")
            after = d.text_after.replace("\n", " ") if d.text_after else ""
            lines.append(f"        before: {before[:MAX_REASON_LENGTH]}")
            if after and after != before:
                lines.append(f"        after:  {after[:MAX_REASON_LENGTH]}")
        elif d.reason:
            lines.append(f"        reason: {d.reason[:MAX_REASON_LENGTH]}")
    return "\n".join(lines)


def write_report(report: str, path: Path) -> None:
    Path(path).write_text(report, encoding="utf-8")
