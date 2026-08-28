from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

from rich.console import Console

from cleancut.config import Config
from cleancut.constants import DEFAULT_SCENE_THRESHOLD, MAX_REASON_LENGTH
from cleancut.edl import EditDecisionList
from cleancut.pipeline import PipelineOptions, build_edl, run_full
from cleancut.probe import (
    audio_streams,
    find_sidecar_subtitle,
    pick_embedded_subtitle,
    probe_streams,
    subtitle_streams,
)
from cleancut.report import build_plan, build_results_report, write_report

console = Console()

CATEGORIES = ("profanity", "drugs", "sex", "violence", "nudity")

# Simple arg-to-config mappings: (argparse_dest, config_attribute).
# These are pure pass-through — if the arg value is not None (or truthy), copy it directly.
_SIMPLE_ARG_MAP: list[tuple[str, str]] = [
    ("whisper_model", "whisper_model"),
    ("whisper_device", "whisper_device"),
    ("whisper_language", "whisper_language"),
    ("visual_threshold", "visual_threshold"),
    ("visual_sample_seconds", "visual_sample_seconds"),
    ("visual_min_streak", "visual_min_streak"),
    ("visual_shot_hit_fraction", "visual_shot_hit_fraction"),
    ("scene_threshold", "scene_threshold"),
    ("encoder", "encoder"),
    ("quality", "quality"),
    ("density", "density_enabled"),
    ("density_window", "density_window_seconds"),
    ("density_min_events", "density_min_events"),
    ("llm", "llm_enabled"),
    ("llm_model", "llm_model"),
    ("llm_host", "llm_host"),
    ("llm_min_confidence", "llm_min_confidence"),
    ("vlm", "vlm_enabled"),
    ("vlm_model", "vlm_model"),
    ("vlm_mode", "vlm_mode"),
    ("vlm_stride", "vlm_stride"),
    ("vlm_min_confidence", "vlm_min_confidence"),
    ("vlm_gaps_radius", "vlm_gaps_radius"),
    ("audio_events", "audio_events_enabled"),
    ("audio_events_threshold", "audio_events_threshold"),
    ("corroboration_radius", "corroboration_radius_seconds"),
]


def _apply_common(args: argparse.Namespace, config: "Config") -> None:
    # Preset goes first so per-flag overrides take precedence.
    if getattr(args, "preset", None):
        config.apply_preset(args.preset)

    # Simple pass-through mappings.
    for arg_name, cfg_attr in _SIMPLE_ARG_MAP:
        val = getattr(args, arg_name, None)
        if val is not None:
            setattr(config, cfg_attr, val)

    # Special transformations that can't be expressed as simple pass-through.
    if args.wordlists:
        config.override_wordlists(Path(args.wordlists))
    if args.replacements:
        config.override_replacements(Path(args.replacements))
    if args.disable_category:
        for cat in args.disable_category:
            config.enabled_categories.discard(cat)
    if args.enable_category:
        for cat in args.enable_category:
            config.enabled_categories.add(cat)
    for spec in args.action or []:
        if "=" not in spec:
            raise SystemExit(f"--action must be CATEGORY=ACTION, got {spec!r}")
        cat, action = spec.split("=", 1)
        if cat not in CATEGORIES:
            raise SystemExit(
                f"unknown category {cat!r}; choose from {'|'.join(CATEGORIES)}"
            )
        if action not in ("mute", "cut", "keep"):
            raise SystemExit(f"action must be mute|cut|keep, got {action!r}")
        config.actions[cat] = action  # type: ignore[assignment]
    if args.no_word_timestamps:
        config.whisper_word_timestamps = False
    if args.no_snap_to_scenes:
        config.snap_cuts_to_scenes = False
    if args.vlm_cut_intimate:
        config.vlm_cut_intimate = True
    if args.allow_solo_visual:
        config.require_visual_corroboration = False
    # Track / language selection is on the PipelineOptions, not Config.


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--preset", choices=["fast", "balanced", "thorough"], default="thorough",
        help="Tuning preset. thorough = best quality (default on capable hardware).",
    )
    p.add_argument("--wordlists", help="Path to wordlists JSON (overrides default).")
    p.add_argument("--replacements", help="Path to replacements JSON (overrides default).")
    p.add_argument(
        "--enable-category", action="append",
        choices=["profanity", "drugs", "sex", "violence", "nudity"],
        help="Enable a category (repeatable).",
    )
    p.add_argument(
        "--disable-category", action="append",
        choices=["profanity", "drugs", "sex", "violence", "nudity"],
        help="Disable a category (repeatable).",
    )
    p.add_argument(
        "--action", action="append",
        metavar="CATEGORY=ACTION",
        help="Override action for a category, e.g. profanity=mute (repeatable).",
    )
    p.add_argument(
        "--whisper-model", default=None,
        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
        help="Whisper model size.",
    )
    p.add_argument(
        "--whisper-device", default=None, choices=["cpu", "mps", "cuda"],
        help="Force Whisper device. Default: autodetect (MPS on Apple Silicon).",
    )
    p.add_argument("--whisper-language", default=None, help="Force language hint (e.g. 'en').")
    p.add_argument("--no-word-timestamps", action="store_true",
                   help="Disable Whisper word-level timestamps (faster, less precise mutes).")
    p.add_argument("--visual-threshold", type=float, default=None, help="NudeNet confidence threshold (0-1).")
    p.add_argument("--visual-sample-seconds", type=float, default=None, help="Sample 1 frame every N seconds.")
    p.add_argument("--visual-min-streak", type=int, default=None,
                   help="Streak mode: consecutive flagged samples needed to emit a cut.")
    p.add_argument("--visual-shot-hit-fraction", type=float, default=None,
                   help="Shot-aware mode: fraction of sampled frames in a shot that must hit (0-1).")
    p.add_argument("--scene-threshold", type=float, default=None,
                   help="PySceneDetect ContentDetector threshold. Lower = more cuts.")
    p.add_argument("--no-snap-to-scenes", action="store_true",
                   help="Don't snap visual/cut ranges to shot boundaries.")
    p.add_argument("--no-visual", action="store_true", help="Skip the NudeNet visual scan.")
    p.add_argument("--no-whisper", action="store_true", help="Don't transcribe if no .srt is found.")
    p.add_argument("--no-scenes", action="store_true", help="Skip PySceneDetect shot boundary detection.")
    p.add_argument("--no-burn-subs", action="store_true", help="Don't burn the softened subs into the video.")
    p.add_argument("--soft-subs", action="store_true",
                   help="Add the softened subs as a toggleable track instead of burning them in.")
    p.add_argument("--density", dest="density", action="store_true", default=None,
                   help="Enable density clustering (default: on for thorough preset).")
    p.add_argument("--no-density", dest="density", action="store_false", default=None,
                   help="Disable density clustering.")
    p.add_argument("--density-window", type=float, default=None,
                   help="Rolling window (seconds) for density clustering. Default 60.")
    p.add_argument("--density-min-events", type=int, default=None,
                   help="Minimum hits in window to count as a cluster. Default 3.")
    p.add_argument("--use-llm", dest="llm", action="store_true", default=None,
                   help="Enable local LLM contextual dialogue classification (via Ollama).")
    p.add_argument("--no-llm", dest="llm", action="store_false", default=None,
                   help="Disable LLM dialogue classification.")
    p.add_argument("--llm-model", default=None,
                   help="Ollama model name (default: llama3.1:8b).")
    p.add_argument("--llm-host", default=None,
                   help="Ollama host URL (default: http://127.0.0.1:11434).")
    p.add_argument("--llm-min-confidence", type=float, default=None,
                   help="Discard LLM classifications below this confidence (0-1). Default 0.6.")
    p.add_argument("--use-vlm", dest="vlm", action="store_true", default=None,
                   help="Enable local VLM scene classification (visual gap-closing).")
    p.add_argument("--no-vlm", dest="vlm", action="store_false", default=None,
                   help="Disable VLM scene classification.")
    p.add_argument("--vlm-model", default=None,
                   help="Ollama vision model (default: llava:7b).")
    p.add_argument("--vlm-mode", default=None,
                   choices=["all", "silent", "gaps", "silent+gaps"],
                   help="Which shots to scan. silent+gaps (default) = silent shots + shots near flagged ranges.")
    p.add_argument("--vlm-stride", type=int, default=None,
                   help="For 'all' mode: scan every Nth shot. Default 1.")
    p.add_argument("--vlm-min-confidence", type=float, default=None,
                   help="Discard VLM classifications below this confidence (0-1). Default 0.55.")
    p.add_argument("--vlm-cut-intimate", action="store_true",
                   help="Also cut shots VLM labels 'intimate' (kissing/undressing, no nudity). Off by default.")
    p.add_argument("--vlm-gaps-radius", type=float, default=None,
                   help="In 'gaps' mode: scan shots within N seconds of a flagged range. Default 30.")
    p.add_argument("--use-audio-events", dest="audio_events", action="store_true", default=None,
                   help="Enable AST-based audio event detection (moans, screams, gunshots).")
    p.add_argument("--no-audio-events", dest="audio_events", action="store_false", default=None)
    p.add_argument("--audio-events-threshold", type=float, default=None,
                   help="AST confidence threshold (0-1). Default 0.45.")
    p.add_argument("--allow-solo-visual", action="store_true",
                   help="Don't require corroboration for visual-only cuts (NudeNet, VLM).")
    p.add_argument("--corroboration-radius", type=float, default=None,
                   help="Visual cuts need a dialogue/audio event within ±N seconds (default 5).")
    p.add_argument("--encoder", default=None, choices=["auto", "videotoolbox", "libx264"],
                   help="Video encoder. auto = videotoolbox on macOS, libx264 elsewhere.")
    p.add_argument("--quality", type=int, default=None,
                   help="Quality (libx264 CRF; lower=better). Default depends on preset.")
    p.add_argument("--audio-track", type=int, default=None,
                   help="0-indexed audio track to transcribe (audio:0, audio:1, …). Default: prefer English.")
    p.add_argument("--prefer-language", default="eng",
                   help="ISO-639 language code to prefer for subs and audio (default: eng).")
    p.add_argument("--save-transcript", default=None,
                   help="Persist Whisper output to this .srt path (and .words.json alongside).")


def _options_from_args(args: argparse.Namespace, **overrides) -> PipelineOptions:
    """Build PipelineOptions from the CLI flags shared by scan/clean."""
    fields: dict = dict(
        video=Path(args.video),
        subs=Path(args.subs) if args.subs else None,
        use_visual=not args.no_visual,
        use_whisper=not args.no_whisper,
        use_scenes=not args.no_scenes,
        audio_track=args.audio_track,
        prefer_language=args.prefer_language,
        save_transcript=Path(args.save_transcript) if args.save_transcript else None,
    )
    fields.update(overrides)
    return PipelineOptions(**fields)


def _write_results_report(
    video: Path | None, output: Path | None, edl: EditDecisionList, report_path: Path,
) -> None:
    """Probe duration (best-effort), then build and write the results report."""
    duration = None
    if video is not None:
        try:
            from cleancut.editor import probe_duration
            duration = probe_duration(video)
        except Exception:
            duration = None
    report = build_results_report(
        video if video is not None else Path("?"), output, edl, original_duration=duration,
    )
    write_report(report, report_path)
    console.print(f"[green]Wrote report[/green] {report_path}")


def cmd_scan(args) -> int:
    config = Config.load_defaults()
    _apply_common(args, config)
    opts = _options_from_args(
        args,
        edl_out=Path(args.output) if args.output else Path(args.video).with_suffix(".edl.json"),
    )
    edl, _ = build_edl(opts, config)
    edl.to_json(opts.edl_out)
    console.print(f"[green]Wrote EDL[/green] {opts.edl_out}")
    console.print(f"[bold]Summary[/bold]: {edl.summary()}")

    # Always write a human-readable report next to the EDL.
    _write_results_report(opts.video, None, edl, opts.edl_out.with_suffix(".report.txt"))
    return 0


def cmd_clean(args) -> int:
    config = Config.load_defaults()
    _apply_common(args, config)
    video = Path(args.video)
    opts = _options_from_args(
        args,
        output=Path(args.output) if args.output else video.with_name(f"{video.stem}.clean{video.suffix}"),
        edl_in=Path(args.edl) if args.edl else None,
        edl_out=Path(args.edl_out) if args.edl_out else None,
        burn_subs=not args.no_burn_subs,
        soft_subs=args.soft_subs,
    )
    out, edl = run_full(opts, config)
    console.print(f"[green bold]Wrote[/green bold] {out}")

    # Report reuses the EDL run_full just built — never re-run detection for it.
    _write_results_report(opts.video, opts.output, edl, opts.output.with_suffix(".report.txt"))
    return 0


def cmd_add_cut(args) -> int:
    from cleancut.edl_ops import fmt_timestamp, parse_timestamp
    from cleancut.scenes import detect_shots

    edl_path = Path(args.edl)
    if not edl_path.exists():
        console.print(f"[red]EDL not found:[/red] {edl_path}")
        return 1
    edl = EditDecisionList.from_json(edl_path)
    start = parse_timestamp(args.start)
    end = parse_timestamp(args.end)
    if end <= start:
        console.print("[red]end must be > start[/red]")
        return 1

    if args.snap:
        try:
            video = Path(edl.video_path) if edl.video_path else None
            if video and video.exists():
                shots = detect_shots(video, threshold=DEFAULT_SCENE_THRESHOLD)
                from cleancut.scenes import snap_range_to_shots
                ns, ne = snap_range_to_shots(start, end, shots)
                console.print(
                    f"[cyan]Snapped to shots:[/cyan] {fmt_timestamp(start)}-{fmt_timestamp(end)} "
                    f"→ {fmt_timestamp(ns)}-{fmt_timestamp(ne)}"
                )
                start, end = ns, ne
        except Exception as e:
            console.print(f"[yellow]Snap failed: {e}[/yellow]")

    from cleancut.edl import EditDecision
    edl.add(EditDecision(
        start=start, end=end,
        action=args.action, category=args.category,
        reason=f"manual: {args.reason}" if args.reason else "manual",
        source="manual",
    ))
    edl = edl.sorted().merge_overlapping(gap=0.5)
    edl.to_json(edl_path)
    console.print(
        f"[green]Added {args.action} {fmt_timestamp(start)}-{fmt_timestamp(end)} "
        f"({args.category}) to[/green] {edl_path}"
    )

    # Refresh report
    _write_results_report(
        Path(edl.video_path) if edl.video_path else None,
        None, edl, edl_path.with_suffix(".report.txt"),
    )
    return 0


def cmd_review(args) -> int:

    edl_path = Path(args.edl)
    if not edl_path.exists():
        console.print(f"[red]EDL not found:[/red] {edl_path}")
        return 1
    edl = EditDecisionList.from_json(edl_path)
    video_str = args.video or edl.video_path
    if not video_str:
        console.print("[red]EDL has no video path — pass --video.[/red]")
        return 1
    video = Path(video_str)
    if not video.exists():
        console.print(f"[red]Video not found:[/red] {video}")
        return 1

    # Optional dialogue context from a .srt
    subs = []
    srt = Path(args.subs) if args.subs else None
    if srt and srt.exists():
        from cleancut.subtitles import read_srt
        subs = read_srt(srt)

    # What to review.
    from cleancut.edl_ops import cuts_for_review
    cuts = cuts_for_review(edl, include_violence=args.include_violence)
    if not cuts:
        console.print("[yellow]No cuts to review.[/yellow]")
        return 0

    _tmp_dir: str | None = None
    if args.frames_dir:
        out_dir = Path(args.frames_dir)
    else:
        _tmp_dir = tempfile.mkdtemp(prefix="cleancut_review_")
        out_dir = Path(_tmp_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        console.print(
            f"[bold]Reviewing {len(cuts)} cut(s)[/bold]   "
            f"(commands: [green]y[/green]=keep, [red]n[/red]=reject, "
            f"[cyan]t START END[/cyan]=trim, [magenta]s[/magenta]=skip, "
            f"[yellow]q[/yellow]=save and quit, [white]o[/white]=open frame)"
        )

        try:
            _review_loop(cuts, video, subs, out_dir)
        except (EOFError, KeyboardInterrupt):
            # Closed stdin / Ctrl-C must not throw away the review work done
            # so far — save and exit cleanly.
            console.print("\n[yellow]Interrupted — saving progress.[/yellow]")
        _save_review(edl, edl_path)
        return 0
    finally:
        if _tmp_dir is not None:
            shutil.rmtree(_tmp_dir, ignore_errors=True)


class _ReviewQuit(Exception):
    """Raised when the user enters 'q' — save and exit."""


def _review_loop(cuts, video: Path, subs: list, out_dir: Path) -> None:
    """Walk the user through each cut, mutating decisions in place."""
    import subprocess
    from cleancut.edl_ops import fmt_timestamp

    try:
        for i, d in enumerate(cuts, 1):
            # Extract frame
            from cleancut.edl_ops import fmt_ffmpeg_timestamp
            ts = fmt_ffmpeg_timestamp(d.start + d.duration / 2.0)
            frame = out_dir / f"cut_{i:02d}.jpg"
            subprocess.run(
                ["ffmpeg", "-y", "-v", "error", "-ss", ts, "-i", str(video),
                 "-frames:v", "1", "-q:v", "3", str(frame)],
                check=False,
            )
            # Dialogue
            lines = []
            for s in subs:
                if s.start < d.end and s.end > d.start:
                    m_ = int(s.start // 60)
                    s_ = s.start - m_ * 60
                    lines.append(f"      [{m_}:{s_:05.2f}] {s.text.strip()}")

            console.print()
            console.print(
                f"[bold]Cut {i}/{len(cuts)}[/bold]  "
                f"{fmt_timestamp(d.start)} → {fmt_timestamp(d.end)}  "
                f"({d.duration:.1f}s)  [cyan]{d.category}[/cyan]  "
                f"[white]via {d.source}[/white]"
            )
            console.print(f"  reason: {d.reason[:MAX_REASON_LENGTH]}")
            if lines:
                console.print("  dialogue:")
                for ln in lines[:10]:
                    console.print(f"  {ln}")
                if len(lines) > 10:
                    console.print(f"      … ({len(lines) - 10} more lines)")
            console.print(f"  frame:    {frame}")

            while True:
                ans = input("  > ").strip().lower()
                if not ans:
                    continue
                if ans == "y":
                    break
                if ans == "n":
                    d.accepted = False
                    console.print("  [red]rejected[/red]")
                    break
                if ans == "s":
                    console.print("  [yellow]skipped (no change)[/yellow]")
                    break
                if ans == "o":
                    subprocess.run(["open", str(frame)], check=False)
                    continue
                if ans == "q":
                    raise _ReviewQuit
                if ans.startswith("t "):
                    try:
                        parts = ans.split(maxsplit=2)
                        if len(parts) < 3:
                            console.print("[red]Expected: t START END[/red]")
                            continue
                        _, new_s, new_e = parts
                        from cleancut.edl_ops import parse_timestamp
                        # Parse both before assigning either, so a bad END
                        # can't leave a half-applied trim.
                        start_val = parse_timestamp(new_s)
                        end_val = parse_timestamp(new_e)
                        if end_val <= start_val:
                            console.print("  [red]end must be > start[/red]")
                            continue
                        d.start, d.end = start_val, end_val
                        console.print(
                            f"  [cyan]trimmed → "
                            f"{fmt_timestamp(d.start)} → {fmt_timestamp(d.end)}[/cyan]"
                        )
                        break
                    except Exception as e:
                        console.print(f"  [red]bad trim: {e}[/red]")
                        continue
                console.print("  [yellow]?[/yellow] y/n/t START END/s/o/q")
    except _ReviewQuit:
        pass


def _save_review(edl: EditDecisionList, edl_path: Path) -> None:
    edl.to_json(edl_path)
    console.print(f"\n[green]Saved {edl_path}[/green]")
    _write_results_report(
        Path(edl.video_path) if edl.video_path else None,
        None, edl, edl_path.with_suffix(".report.txt"),
    )


def cmd_inspect(args) -> int:
    video = Path(args.video)
    if not video.exists():
        console.print(f"[red]Not found:[/red] {video}")
        return 1
    config = Config.load_defaults()
    _apply_common(args, config)

    streams = probe_streams(video)
    console.print(f"[bold]File:[/bold] {video}")

    audios = audio_streams(streams)
    console.print(f"\n[bold cyan]Audio tracks ({len(audios)})[/bold cyan]")
    for i, s in enumerate(audios):
        ch = f"{s.channels}ch" if s.channels else ""
        console.print(f"  audio:{i}  index={s.index}  lang={s.language}  {s.codec_name}  {ch}  {s.title}")

    subs_streams = subtitle_streams(streams)
    console.print(f"\n[bold cyan]Embedded subtitle tracks ({len(subs_streams)})[/bold cyan]")
    from cleancut.probe import TEXT_SUBTITLE_CODECS
    for i, s in enumerate(subs_streams):
        kind = "TEXT" if s.codec_name in TEXT_SUBTITLE_CODECS else "image"
        console.print(f"  sub:{i}    index={s.index}  lang={s.language}  {s.codec_name} [{kind}]  {s.title}")
    embedded = pick_embedded_subtitle(streams, prefer_language=args.prefer_language)
    if embedded:
        console.print(f"  → would extract: stream index {embedded.index} ({embedded.codec_name}, {embedded.language})")

    sidecar = find_sidecar_subtitle(video, prefer_language=args.prefer_language)
    console.print("\n[bold cyan]Sidecar .srt files[/bold cyan]")
    if sidecar:
        console.print(f"  → chosen: {sidecar}")
    for p in sorted(video.parent.glob("*.srt")):
        marker = " ✓" if p == sidecar else ""
        console.print(f"  {p.name}{marker}")

    # The plan — "what would happen if I ran clean now"
    console.print()
    plan = build_plan(
        video=video,
        streams=streams,
        config=config,
        audio_track=args.audio_track,
        prefer_language=args.prefer_language,
        use_visual=not args.no_visual,
        use_scenes=not args.no_scenes,
        use_whisper=not args.no_whisper,
        burn_subs=not args.no_burn_subs,
        explicit_subs=Path(args.subs) if args.subs else None,
    )
    console.print(plan)
    if args.report_out:
        write_report(plan, Path(args.report_out))
        console.print(f"\n[green]Wrote plan[/green] {args.report_out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cleancut", description="Auto-edit movies for content.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_scan = sub.add_parser("scan", help="Build an EDL without rendering.")
    p_scan.add_argument("video", help="Input video.")
    p_scan.add_argument("--subs", help="External .srt file (else look beside video, else Whisper).")
    p_scan.add_argument("-o", "--output", help="EDL output path (default: <video>.edl.json).")
    _add_common(p_scan)
    p_scan.set_defaults(func=cmd_scan)

    p_clean = sub.add_parser("clean", help="Build an EDL and render the cleaned video.")
    p_clean.add_argument("video", help="Input video.")
    p_clean.add_argument("--subs", help="External .srt file.")
    p_clean.add_argument("-o", "--output", help="Output video (default: <video>.clean.<ext>).")
    p_clean.add_argument("--edl", help="Use an existing EDL JSON instead of scanning.")
    p_clean.add_argument("--edl-out", help="Also write the (re-)built EDL here.")
    _add_common(p_clean)
    p_clean.set_defaults(func=cmd_clean)

    p_inspect = sub.add_parser("inspect", help="Show tracks + a plan of what cleancut would change.")
    p_inspect.add_argument("video", help="Input video.")
    p_inspect.add_argument("--subs", help="External .srt file (overrides discovery).")
    p_inspect.add_argument("--report-out", help="Write the plan to this text file.")
    _add_common(p_inspect)
    p_inspect.set_defaults(func=cmd_inspect)

    p_add = sub.add_parser("add-cut", help="Add a manual cut/mute to an existing EDL.")
    p_add.add_argument("edl", help="EDL JSON file to modify.")
    p_add.add_argument("--start", required=True, help="Start timestamp (MM:SS or H:MM:SS or seconds).")
    p_add.add_argument("--end", required=True, help="End timestamp.")
    p_add.add_argument("--action", choices=["cut", "mute"], default="cut")
    p_add.add_argument("--category", default="manual")
    p_add.add_argument("--reason", default=None)
    p_add.add_argument("--snap", action="store_true", help="Snap range outward to nearest shot boundaries.")
    p_add.set_defaults(func=cmd_add_cut)

    p_rev = sub.add_parser("review", help="Interactively approve/reject/trim cuts in an EDL.")
    p_rev.add_argument("edl", help="EDL JSON file to review.")
    p_rev.add_argument("--video", default=None, help="Source video (default: from EDL).")
    p_rev.add_argument("--subs", default=None, help="External .srt for dialogue context.")
    p_rev.add_argument("--frames-dir", default=None, help="Where to extract preview frames (default: /tmp).")
    p_rev.add_argument("--include-violence", action="store_true",
                       help="Also review pure-violence cuts (off by default — fights kept).")
    p_rev.set_defaults(func=cmd_review)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        console.print("[yellow]Interrupted[/yellow]")
        return 130
    except Exception as e:
        console.print(f"[red bold]error[/red bold]: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
