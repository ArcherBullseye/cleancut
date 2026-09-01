# Local changes to the upstream `cleancut` package

This fork vendors [monahand1023/cleancut](https://github.com/monahand1023/cleancut)
so upstream fixes can be pulled in cleanly. The detection pipeline is unmodified.
Six changes have been made, all recorded here.

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

## 4. Subtitle softener covers the detector's inflections (subtitles.py) — bug fix

The wordlists spell open suffixes (`\w*`), so the detector matches "porno",
"fuckers", "raped". `replacements.json` holds base forms only and `soften_text`
matched them as exact `\b`-anchored literals, so those words were muted or cut
in the audio while the softened subtitle track still displayed them in full — a
kid reading subtitles saw what the audio had removed.

`soften_text` now takes an optional `wordlists` argument and folds the
detector's `\w*` patterns into the same single-pass alternation, resolving each
match to the longest base-form key that prefixes it. A match with no such key is
returned untouched rather than mangled.

Only patterns spelling `\w*` are borrowed. That matters: `ass` is a replacement
key, and widening every key by a suffix would rewrite "assistant" and "assume".
The detector has no bare-`ass` pattern, so keying off its patterns is safe —
covered by tests.

## 5. Detectors respect the configured actions (four files) — bug fix

`subtitles.py` and `visual.py` resolved each hit through
`config.actions.get(category, ...)`, but the other four detectors hardcoded
`action="cut"`:

    classify_dialogue.py:223   audio_events.py:236
    classify_visual.py:268     density.py:91

So a scan run with `violence=keep` still cut every violent scene the LLM found,
and categories set to `mute` were cut instead. That is not just a correctness
problem: a cut forces a full video re-encode, while a mute-only edit
stream-copies, so the bug also turned minute-long renders into multi-hour ones.

`resolve_action()` in `edl.py` now maps a (possibly composite) category to the
configured action. Composite categories like `violence+profanity` take the
strongest action any component asks for, and are dropped only when every
component says keep. Each detector's params dataclass gained an `actions` field,
defaulting to `None`, which preserves the old cut-everything behaviour for
direct callers; `pipeline.py` passes `config.actions` at all four sites.

Note `density.py` uses a guard rather than `continue` -- the loop's `i = k + 1`
must still run.

## 6. macOS-compatible MP4 output (editor.py, probe.py, review.py) — bug fix

H.264 encoders inherited the source pixel format, so a 10-bit or 4:4:4 input
could produce a valid output that QuickTime and Safari could not decode. MP4
metadata was also written at the end of the file, and copied HEVC used ffmpeg's
`hev1` tag instead of Apple's expected `hvc1` tag.

All H.264 encodes and browser review clips now emit 8-bit `yuv420p`. MP4 output
uses `+faststart`, copied HEVC is tagged `hvc1`, and known-incompatible source
video formats are converted to H.264. Compatible H.264 and HEVC sources retain
the fast stream-copy path used by mute-only renders.

Upstream's test suite passes with regression coverage for all six changes.
