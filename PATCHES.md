# Local changes to the upstream `cleancut` package

This fork vendors [monahand1023/cleancut](https://github.com/monahand1023/cleancut)
so upstream fixes can be pulled in cleanly. The detection pipeline is unmodified.
Exactly one additive change has been made:

## `--soft-subs` (pipeline.py, cli.py)

Upstream writes the softened subtitles by **burning them into the picture**
whenever ffmpeg has libass — which Debian's ffmpeg does, so the container always
hit that path. Burned-in subtitles are irreversible and permanently visible,
which is the wrong default for a movie you keep in a library.

The change adds a `soft_subs` field to `PipelineOptions` and a `--soft-subs`
CLI flag. When set, the softened `.srt` is muxed into the container as a
toggleable track instead of being painted onto the frames. `apply_mutes_and_subs`
already supported this — the soft-sub branch was simply unreachable from the CLI.

Three call sites, all additive; no existing behaviour changes when the flag is
absent. Upstream's own test suite (283 tests) passes unmodified.

To pull upstream changes: replace `cleancut/` with the new upstream copy, then
re-apply this flag.
