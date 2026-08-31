# Local changes to the upstream `cleancut` package

This fork vendors [monahand1023/cleancut](https://github.com/monahand1023/cleancut)
so upstream fixes can be pulled in cleanly. The detection pipeline is unmodified.
Three changes have been made, all recorded here.

To pull upstream changes: replace `cleancut/` with the new upstream copy, then
re-apply each of these.

## 1. `--soft-subs` (pipeline.py, cli.py) — additive

Upstream writes the softened subtitles by **burning them into the picture**
whenever ffmpeg has libass — which Debian's ffmpeg does, so the container always
hit that path. Burned-in subtitles are irreversible and permanently visible,
which is the wrong default for a movie you keep in a library.

The change adds a `soft_subs` field to `PipelineOptions` and a `--soft-subs`
CLI flag. When set, the softened `.srt` is muxed into the container as a
toggleable track instead of being painted onto the frames. `apply_mutes_and_subs`
already supported this — the soft-sub branch was simply unreachable from the CLI.

Three call sites, all additive; no existing behaviour changes when the flag is
absent.

## 2. Surround-audio channel layout (editor.py) — bug fix

ffmpeg's native AAC encoder refuses to open when the channel layout is
unspecified, reporting the input as "6 channels" rather than "5.1". A DDP 5.1
source whose container carries no layout tag decodes to exactly that, so every
5.1 file — the norm for WEB-DL — died at the final mux with "Unsupported channel
layout", *after* the video had already been encoded.

Both audio encode paths now end their filter chain in an `aformat` pin
(`KNOWN_LAYOUTS`), and `_audio_encoder_args` scales the AAC bitrate with the
source channel count; the previous flat 192k gave 5.1 about 32k per channel.

Stereo sources were unaffected, which is why upstream's suite never caught it.

## 3. Exit code from `python -m cleancut` (__main__.py) — bug fix

`__main__.py` called `main()` and discarded its return value, so the process
exited 0 even when `main()` returned 1 after an error. Anything driving the CLI
as a subprocess — which is exactly what `webapp/jobs.py` does — saw every
failure as success. A render that died in ffmpeg was recorded as Complete with a
zero-byte output file.

Now `sys.exit(main())`. Upstream's own `cli.py` already does this under its
`if __name__ == "__main__"` guard, which never fires under `-m`.

Upstream's test suite passes unmodified against all three.
