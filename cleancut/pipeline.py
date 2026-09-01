"""Full pipeline: video + subs -> EDL -> cleaned video."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console

from cleancut.config import Config
from cleancut.edl import EditDecisionList, snap_edl_to_shots
from cleancut.editor import (
    adjust_subtitles_for_cuts,
    apply_cuts,
    apply_mutes_and_subs,
    edl_to_ranges,
    shift_ranges_after_cuts,
)
from cleancut.scenes import Shot
from cleancut.subtitles import (
    Subtitle,
    read_srt,
    scan_subtitles,
    scan_words,
    softened_subtitles,
    write_srt,
)
from cleancut.transcribe import Word

console = Console()


def _load_words_sidecar(srt_path: Path) -> list[Word]:
    """Load the .words.json written by --save-transcript, if one sits next to
    the .srt — restores word-level mute precision on reruns with --subs."""
    import json

    words_path = srt_path.with_suffix(".words.json")
    if not words_path.exists():
        return []
    try:
        data = json.loads(words_path.read_text())
        words = [Word(**w) for w in data]
    except (json.JSONDecodeError, TypeError):
        return []
    if words:
        console.print(f"[cyan]Loaded word timings[/cyan] {words_path}")
    return words


def _run_detector(name: str, fn, *args, **kwargs) -> list:
    """Run a detector, returning [] and printing a warning on any error.

    Detectors raise RuntimeError for expected problems (missing extras, Ollama
    down), but unexpected errors (cv2, transport) must not abort a scan that is
    hours in — warn and move on.
    """
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        console.print(f"[yellow]{name} skipped: {e}[/yellow]")
        return []


# Backward-compatible alias — test_pipeline_unit.py imports this from pipeline.
_snap_edl_to_shots = snap_edl_to_shots


@dataclass
class PipelineOptions:
    video: Path
    subs: Path | None = None
    output: Path | None = None
    edl_in: Path | None = None
    edl_out: Path | None = None
    use_visual: bool = True
    use_whisper: bool = True
    use_scenes: bool = True
    burn_subs: bool = True
    # 0-indexed audio track ordinal (audio:0, audio:1, …); None = auto (prefer English).
    audio_track: int | None = None
    # ISO language code to prefer for both sidecar .srt and audio track ("eng", "spa", …).
    prefer_language: str = "eng"
    # If set, persist Whisper output to this .srt path (and .words.json alongside).
    save_transcript: Path | None = None
    # Write the softened subtitles as a toggleable track in the container
    # instead of burning them into the picture. Burn-in is irreversible, which
    # is the wrong default for a media library you keep.
    soft_subs: bool = False


def _get_subtitles_and_words(
    opts: PipelineOptions, config: Config
) -> tuple[list[Subtitle], list]:
    """Resolve subtitles in this priority:

    1. Explicit --subs path.
    2. Best sidecar .srt found beside the video (Plex naming, language-aware).
    3. Best embedded *text* subtitle track in the container.
    4. Whisper transcription of the preferred-language audio track.
    """
    from cleancut.probe import (
        extract_audio_to_wav,
        extract_text_subtitle,
        find_sidecar_subtitle,
        pick_audio_track,
        pick_embedded_subtitle,
        probe_streams,
    )

    if opts.subs:
        if not opts.subs.exists():
            # A typo'd --subs path must not silently fall through to an
            # hours-long Whisper transcription.
            raise ValueError(f"--subs file not found: {opts.subs}")
        console.print(f"[cyan]Reading subtitles[/cyan] {opts.subs}")
        return read_srt(opts.subs), _load_words_sidecar(opts.subs)

    sidecar = find_sidecar_subtitle(opts.video, prefer_language=opts.prefer_language)
    if sidecar:
        console.print(f"[cyan]Found sidecar .srt[/cyan] {sidecar}")
        return read_srt(sidecar), _load_words_sidecar(sidecar)

    streams = probe_streams(opts.video)
    embedded = pick_embedded_subtitle(streams, prefer_language=opts.prefer_language)
    if embedded:
        console.print(
            f"[cyan]Extracting embedded {embedded.codec_name} subtitle "
            f"(lang={embedded.language})[/cyan]"
        )
        extracted = extract_text_subtitle(opts.video, embedded.index)
        if extracted:
            return read_srt(extracted), []

    if not opts.use_whisper:
        console.print("[yellow]No subtitles and Whisper disabled — skipping dialogue scan.[/yellow]")
        return [], []

    # Pick which audio track to feed Whisper. A bad --audio-track raises
    # ValueError, which the CLI's top-level handler reports.
    track = pick_audio_track(streams, opts.audio_track, prefer_language=opts.prefer_language)
    if track is None:
        console.print("[yellow]No audio tracks found — skipping dialogue scan.[/yellow]")
        return [], []

    console.print(
        f"[cyan]Using audio track[/cyan] index={track.index} lang={track.language} "
        f"codec={track.codec_name}"
    )

    # Transcription is the single most expensive stage — cache it like the
    # cheaper detectors (shots, NudeNet, AST) already are.
    from cleancut import cache as _cache

    h = _cache.config_hash(
        model=config.whisper_model,
        word_timestamps=config.whisper_word_timestamps,
        language=config.whisper_language,
        track=track.index,
    )
    hit = _cache.load(opts.video, "whisper", h)
    if hit is not None:
        console.print("[green]Whisper transcript loaded from cache[/green]")
        subs = [Subtitle(**s) for s in hit.get("subs", [])]
        words = [Word(**w) for w in hit.get("words", [])]
    else:
        import dataclasses

        from cleancut.transcribe import _resolve_device, transcribe

        audio_path = extract_audio_to_wav(opts.video, track.index)
        device = _resolve_device(config.whisper_device, config.whisper_word_timestamps)
        console.print(
            f"[cyan]Transcribing with Whisper[/cyan] model={config.whisper_model} "
            f"device={device} word_timestamps={config.whisper_word_timestamps}"
        )
        try:
            subs, words = transcribe(
                opts.video,
                model_name=config.whisper_model,
                device=device,
                word_timestamps=config.whisper_word_timestamps,
                language=config.whisper_language,
                audio_path=audio_path,
            )
        finally:
            audio_path.unlink(missing_ok=True)
        _cache.save(opts.video, "whisper", h, {
            "subs": [dataclasses.asdict(s) for s in subs],
            "words": [dataclasses.asdict(w) for w in words],
        })

    # Persist the transcript so subsequent scans can reuse it without re-running Whisper.
    if opts.save_transcript:
        srt_out = Path(opts.save_transcript)
        srt_out.parent.mkdir(parents=True, exist_ok=True)
        write_srt(subs, srt_out)
        console.print(f"[green]Saved transcript[/green] {srt_out}")
        if words:
            words_out = srt_out.with_suffix(".words.json")
            import json
            words_out.write_text(json.dumps(
                [{"start": w.start, "end": w.end, "text": w.text, "probability": w.probability}
                 for w in words], indent=2,
            ))
            console.print(f"[green]Saved word timings[/green] {words_out}")

    return subs, words


def _detect_scenes_if_enabled(opts: PipelineOptions, config: Config) -> list[Shot]:
    if not opts.use_scenes:
        return []
    try:
        from cleancut.scenes import detect_shots
    except RuntimeError as e:
        console.print(f"[yellow]Scene detection skipped: {e}[/yellow]")
        return []
    console.print(f"[cyan]Detecting shot boundaries[/cyan] (threshold={config.scene_threshold})")
    try:
        shots = detect_shots(opts.video, threshold=config.scene_threshold)
        console.print(f"[green]Found {len(shots)} shots[/green]")
        return shots
    except Exception as e:
        console.print(f"[yellow]Scene detection failed: {e}[/yellow]")
        return []


def build_edl(opts: PipelineOptions, config: Config) -> tuple[EditDecisionList, list[Subtitle]]:
    """Run all detectors and produce a merged EDL."""
    subs, words = _get_subtitles_and_words(opts, config)

    edl = EditDecisionList(video_path=str(opts.video), subtitle_path=str(opts.subs or ""))

    # Dialogue scan: prefer word-level if Whisper gave us words, else line-level.
    if words:
        console.print(f"[cyan]Scanning {len(words)} words (word-precise)[/cyan]")
        edl.extend(scan_words(words, config).decisions)
    elif subs:
        console.print(f"[cyan]Scanning {len(subs)} subtitle lines[/cyan]")
        edl.extend(scan_subtitles(subs, config).decisions)

    # Shot boundaries (also used for shot-aware visual scan below).
    shots = _detect_scenes_if_enabled(opts, config)

    if opts.use_visual and "nudity" in config.enabled_categories:
        from cleancut.visual import scan_video
        console.print(
            f"[cyan]Visual scan[/cyan] "
            f"({'shot-aware' if shots else 'streak mode'}, "
            f"sample={config.visual_sample_seconds}s, "
            f"threshold={config.visual_threshold})"
        )
        visual_decisions = _run_detector(
            "Visual scan", lambda: scan_video(opts.video, config, shots=shots or None).decisions
        )
        edl.extend(visual_decisions)

    edl = edl.pad(config.pad_seconds).merge_overlapping(gap=config.merge_gap_seconds).sorted()

    # LLM-based contextual dialogue classification.
    if config.llm_enabled and subs:
        from cleancut.classify_dialogue import LLMParams, classify_dialogue
        console.print(f"[cyan]LLM dialogue scan[/cyan] model={config.llm_model}")
        llm_params = LLMParams(
            model=config.llm_model,
            ollama_host=config.llm_host,
            min_confidence=config.llm_min_confidence,
            actions=config.actions,
        )
        llm_edl_decisions = _run_detector(
            "LLM scan", lambda: classify_dialogue(subs, llm_params, video=opts.video).decisions
        )
        if llm_edl_decisions:
            console.print(f"[green]LLM flagged {len(llm_edl_decisions)} scene(s)[/green]")
            edl.extend(llm_edl_decisions)

    # Audio event detection (AST) — catches moans/screams/gunshots without dialogue or visible content.
    if config.audio_events_enabled and shots:
        from cleancut import audio_events
        console.print(f"[cyan]Audio event scan[/cyan] model={config.audio_events_model}")
        ae_params = audio_events.AudioEventParams(
            model=config.audio_events_model,
            threshold=config.audio_events_threshold,
            clip_seconds=config.audio_events_clip_seconds,
            skip_violence=config.audio_events_skip_violence,
            actions=config.actions,
        )
        # Analyze the same track Whisper transcribes — ffmpeg's default stream
        # can be a commentary or foreign dub.
        track_index: int | None = None
        try:
            from cleancut.probe import pick_audio_track, probe_streams
            track = pick_audio_track(
                probe_streams(opts.video), opts.audio_track,
                prefer_language=opts.prefer_language,
            )
            track_index = track.index if track else None
        except Exception:
            pass
        ae_edl_decisions = _run_detector(
            "Audio events",
            lambda: audio_events.scan_audio_events(
                opts.video, shots, ae_params, audio_track_index=track_index,
            ).decisions,
        )
        if ae_edl_decisions:
            console.print(f"[green]Audio events flagged {len(ae_edl_decisions)} shot(s)[/green]")
            edl.extend(ae_edl_decisions)

    # Density clustering — promote dense clusters of small hits into a single
    # cut. Runs after the LLM and audio detectors so their decisions (both
    # listed in density.DIALOGUE_SOURCES) can contribute to clusters.
    if config.density_enabled and len(edl) >= config.density_min_events:
        from cleancut.density import DensityParams, find_clusters
        console.print("[cyan]Density clustering[/cyan] over dialogue/audio hits")
        clusters = find_clusters(
            edl,
            DensityParams(
                window_seconds=config.density_window_seconds,
                min_events=config.density_min_events,
                min_cluster_span=config.density_min_cluster_span,
                actions=config.actions,
            ),
        )
        if len(clusters):
            console.print(f"[green]Density found {len(clusters)} cluster(s)[/green]")
            edl.extend(clusters.decisions)

    # VLM visual scene classification — closes the gap on silent scenes.
    if config.vlm_enabled and shots:
        from cleancut.classify_visual import VLMParams, scan_with_vlm as vlm_scan
        cut_on = ("intimate", "explicit", "drug_use", "violence") if config.vlm_cut_intimate \
            else ("explicit", "drug_use", "violence")
        console.print(
            f"[cyan]VLM scene scan[/cyan] model={config.vlm_model} "
            f"mode={config.vlm_mode}"
        )
        vlm_params = VLMParams(
            model=config.vlm_model,
            mode=config.vlm_mode,
            stride=config.vlm_stride,
            min_confidence=config.vlm_min_confidence,
            gaps_radius_seconds=config.vlm_gaps_radius,
            cut_on=cut_on,
            ollama_host=config.llm_host,
            actions=config.actions,
        )
        vlm_edl_decisions = _run_detector(
            "VLM scan",
            lambda: vlm_scan(opts.video, shots, subs, edl, vlm_params).decisions,
        )
        if vlm_edl_decisions:
            console.print(f"[green]VLM flagged {len(vlm_edl_decisions)} shot(s)[/green]")
            edl.extend(vlm_edl_decisions)

    # Re-merge after adding density/LLM/VLM signals.
    edl = edl.merge_overlapping(gap=config.merge_gap_seconds).sorted()
    if shots and config.snap_cuts_to_scenes:
        edl = _snap_edl_to_shots(edl, shots)
        edl = edl.merge_overlapping(gap=0.0).sorted()

    # Cross-signal corroboration — kill solo-visual flags without dialogue/audio backup.
    if config.require_visual_corroboration:
        from cleancut.corroboration import mark_unsupported_visual
        edl, n_marked = mark_unsupported_visual(
            edl, radius_seconds=config.corroboration_radius_seconds,
        )
        if n_marked:
            console.print(
                f"[yellow]Suppressed {n_marked} solo-visual cut(s)[/yellow] "
                f"(no dialogue/audio within ±{config.corroboration_radius_seconds}s)"
            )

    return edl, subs


def render(
    edl: EditDecisionList,
    subs: list[Subtitle],
    opts: PipelineOptions,
    config: Config,
) -> Path:
    """Apply the EDL to the video. Returns the path to the cleaned video."""
    if not opts.output:
        raise ValueError("output path required for render")

    # Work dir alongside the OUTPUT, not the source — source dir may have
    # weird characters that break ffmpeg filters, or may be on a read-only volume.
    work = opts.output.parent / ".cleancut_work"
    work.mkdir(parents=True, exist_ok=True)

    try:
        cuts = edl_to_ranges(edl, "cut")
        mutes = edl_to_ranges(edl, "mute")
        encoder = config.resolved_encoder()
        console.print(f"[cyan]Encoder[/cyan]: {encoder} (q={config.quality})")

        # Step 1: apply cuts (re-encode if needed).
        if cuts:
            cut_path = work / f"{opts.video.stem}.cut.mp4"
            console.print(f"[cyan]Applying {len(cuts)} cut(s)…[/cyan]")
            apply_cuts(opts.video, cuts, cut_path, encoder=encoder, quality=config.quality)
            mutes = shift_ranges_after_cuts(mutes, cuts)
            subs = adjust_subtitles_for_cuts(subs, cuts)
        else:
            cut_path = opts.video

        # Step 2: soften surviving subs and write a working .srt for burn-in.
        softened = softened_subtitles(subs, config) if subs else []
        srt_for_burn: Path | None = None
        if softened and (opts.burn_subs or opts.soft_subs):
            srt_for_burn = work / f"{opts.video.stem}.softened.srt"
            write_srt(softened, srt_for_burn)

        # Step 3: mute audio ranges and burn the softened subs.
        console.print(
            f"[cyan]Muting {len(mutes)} range(s){' + burning subs' if srt_for_burn else ''}[/cyan]"
        )
        apply_mutes_and_subs(
            input_path=cut_path,
            mutes=mutes,
            srt_path=srt_for_burn,
            output_path=opts.output,
            # burn_subs=False with an srt still muxes it as a soft track.
            burn_subs=bool(srt_for_burn) and not opts.soft_subs,
            encoder=encoder,
            quality=config.quality,
        )
    finally:
        # Intermediates (e.g. the movie-sized .cut.mp4) would otherwise leak
        # gigabytes per run.
        shutil.rmtree(work, ignore_errors=True)

    return opts.output


def run_full(opts: PipelineOptions, config: Config) -> tuple[Path, EditDecisionList]:
    """Run the pipeline and return (output path, the EDL that was applied).

    Returning the EDL lets callers build reports without re-running detection.
    """
    if opts.edl_in:
        console.print(f"[cyan]Loading EDL from[/cyan] {opts.edl_in}")
        edl = EditDecisionList.from_json(opts.edl_in)
        # Subtitles are only needed to burn them in — don't trigger a
        # potential Whisper run for subs that would never be used.
        subs = []
        if opts.burn_subs:
            subs, _ = _get_subtitles_and_words(opts, config)
    else:
        edl, subs = build_edl(opts, config)

    if opts.edl_out:
        edl.to_json(opts.edl_out)
        console.print(f"[green]Wrote EDL[/green] {opts.edl_out}")

    console.print(f"[bold]EDL summary[/bold]: {edl.summary()}")

    if opts.output:
        return render(edl, subs, opts, config), edl
    return opts.edl_out or Path(), edl
