# cleancut for Umbrel

Auto-edit movies for content on your own hardware — mute profanity, cut explicit
scenes, and soften subtitles, with every decision reviewable in the browser
before anything is rendered. Nothing leaves the box.

This packages [monahand1023/cleancut](https://github.com/monahand1023/cleancut)
(MIT, by Dan Monahan) as an Umbrel community app: the upstream detection
pipeline, vendored essentially as-is, behind a web UI and a background job queue.

## What it adds to upstream

- **Library browser** over the Umbrel's network shares and home folder.
- **Background job queue** — scans take hours; jobs run one at a time, survive a
  restart, and can be cancelled mid-run.
- **Browser review** of the edit decision list, the web equivalent of upstream's
  interactive `review` command: a thumbnail and a playable clip for every
  proposed cut, per-decision accept / reject / retime, bulk actions, and manual
  cuts. Edits are written straight back into the EDL the renderer consumes.
- **Toggleable softened subtitles** rather than burned-in ones — see
  [PATCHES.md](PATCHES.md), the only change made to the upstream package.
- Model weights and caches pinned into the app's data directory, so an app
  update does not re-download gigabytes.

## Detection signals

Wordlist and context matching, Whisper speech-to-text when there is no subtitle
track, density clustering, NudeNet frame analysis, Ollama LLM dialogue
classification, LLaVA vision, and HuggingFace AST audio events. Visual-only
detections require a second corroborating signal before they fire, and cuts snap
to shot boundaries.

## Presets

Timings are for a feature film on an Umbrel Home (4-core x86, no GPU).

| Preset | Signals | Rough time |
|---|---|---|
| Fast | Whisper `base`, wordlists | 15–30 min |
| Balanced | Whisper `small` + word timing, density clustering | 1–2 hours |
| Thorough | Everything, incl. Ollama LLM + vision | Most of a day |

Those are **scan** times. Rendering is separate, and its cost depends entirely
on what you accept at review: a mute-only edit stream-copies the video and
finishes in minutes, while a single accepted cut forces a full re-encode of the
whole film — hours on four cores. The review page says which one you are about
to get.

## Ollama

Optional. The LLM and vision signals used by the Thorough preset need the Ollama
app; without it they are skipped and the rest of the pipeline runs normally.
Umbrel puts every app on one bridge network, so the default host is
`http://ollama_ollama_1:11434`. Pull `llama3.1:8b` and `llava:7b` to use them.
Settings → Test connection reports what Ollama actually has.

## If the library looks empty

The app bind-mounts the Umbrel's whole `network` directory, and Docker copies
the network shares mounted underneath it when the container starts. If a share
is mounted (or re-mounted, after a NAS reboot or a network blip) *after* the
container is already running, the container keeps its older view and the folder
looks empty. Restarting the app from the Umbrel dashboard re-creates the mount
and picks the share back up.

## Layout

    cleancut/    vendored upstream package (see PATCHES.md)
    webapp/      Flask server: job queue, library, EDL review
    templates/   pages
    static/      styles and client JS
    tests/       upstream's test suite

## Development

    python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
    PYTHONPATH=. .venv/bin/python -m webapp.app     # http://localhost:3000
    PYTHONPATH=. .venv/bin/python -m pytest tests -q

`DATA_DIR` and `MEDIA_ROOTS` (colon-separated) override where state and media
live; both default to the container's paths.

## Releasing

Bump the version in **three** places, then push:

1. `webapp/app.py` — `APP_VERSION`
2. `umbrel-app.yml` — `version:` and `releaseNotes:`
3. the community store repo's copy of `umbrel-app.yml`

Pushing to `main` builds and pushes `ghcr.io/archerbullseye/cleancut-app:latest`.
The Umbrel app pulls that image; the store repo holds only the two manifests.

## License

Upstream cleancut is MIT, © Dan Monahan — see [LICENSE](LICENSE). The Umbrel
packaging and web UI in `webapp/`, `templates/` and `static/` are under the same
terms.
