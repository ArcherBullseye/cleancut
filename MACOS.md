# CleanCut Mac 2.0 beta

This edition runs the web app and processing pipeline directly on Apple
Silicon. It does not use the Umbrel/Linux container, so FFmpeg can use Apple's
VideoToolbox encoders and PyTorch can use Metal.

## Install

1. Install [Homebrew](https://brew.sh) if it is not already installed.
2. Clone the Mac branch, or update an existing checkout, and run the installer:

   ```sh
   git clone --branch codex/macos-native-v2 https://github.com/ArcherBullseye/cleancut.git "$HOME/cleancut"
   cd "$HOME/cleancut"
   ./macos/install.sh
   ```

   If `$HOME/cleancut` already exists, use this instead:

   ```sh
   cd "$HOME/cleancut"
   git pull origin codex/macos-native-v2
   ./macos/install.sh
   ```

3. Mount the movie share in Finder with **Go → Connect to Server** and its
   `smb://...` address. It will appear below `/Volumes`. Use an account with
   read/write access if cleaned videos should be returned to the share.
4. Start Ollama and make sure the local multimodal model is available:

   ```sh
   ollama pull qwen3.5:9b
   ```

5. Double-click `macos/Start CleanCut.command` and open
   <http://127.0.0.1:3000>.

Do not type `/path/to/cleancut` literally; documentation sometimes uses it as
a placeholder. The commands above install this checkout at `$HOME/cleancut`.

Application state, downloaded models, proxies, logs, and default outputs live
in `~/Library/Application Support/CleanCut`. Source movies are never modified.

## NAS workflow

Finder-mounted SMB/NFS shares under `/Volumes` appear automatically in the
CleanCut library. By default, a source such as
`/Volumes/Movies/Film/Film.mkv` is returned as
`/Volumes/Movies/Film/Film.clean.mp4`.

CleanCut does not encode directly into a network file. It renders and fully
validates the video on the Mac's local disk, uploads it to a hidden temporary
file beside the NAS destination, verifies the uploaded container, and then
renames it into place. A disconnect cannot overwrite the source or expose an
incomplete file as the finished video. If publishing fails, the job reports the
error and retains the verified local render in its job directory.

Keep the share mounted until the job finishes. The Mac also needs enough local
free space for the completed video and, when cuts require it, an intermediate
copy. A custom mounted location can be added with `CLEANCUT_MEDIA_ROOTS`.

## 4K defaults

- Scene, NudeNet, and VLM analysis use a cached 720p proxy. Decisions retain
  source timestamps, while the final render always reads the original video.
- Balanced and Thorough use NudeNet's free 640m model. It downloads once
  (99 MB), is checksum-verified, and thereafter runs fully locally. CleanCut
  requests CoreML on Apple Silicon and automatically falls back to local CPU
  inference if a model/operator is unsupported.
- Possible nudity is rescanned at a higher frame rate. Repeated medium-confidence
  hits or a single strong explicit hit create the cut, so brief or silent nudity
  is no longer lost inside a long shot or rejected for lacking dialogue.
- Local AI defaults to Ollama at `127.0.0.1:11434`. The same `qwen3.5:9b`
  model handles dialogue context and visual scene classification; CleanCut
  requests non-thinking JSON output so scans do not spend time generating
  hidden reasoning. No prompts, frames, or results leave the Mac.
- `Automatic` encoding selects hardware H.264 for SDR and hardware 10-bit HEVC
  for HDR/PQ/HLG input.
- Edited video is normalized to the source's average frame rate. This prevents
  FFmpeg cut filters from writing invalid high codec levels that Apple players
  may reject or play with unstable frame pacing.
- Final output is written to a hidden partial file, validated, and only then
  atomically moved into place.
- Full verification decodes the completed video and audio streams. It adds one
  read pass but catches truncated or corrupt frames before a job is marked done.

Dolby Vision enhancement metadata cannot reliably survive an arbitrary cut and
re-encode. CleanCut preserves ordinary HDR10/HLG color descriptions; Dolby
Vision sources should be reviewed as HDR10-compatible output after rendering.

## Moving to another Mac

Clone the same repository and run `macos/install.sh` on the destination Mac.
To retain job history and model caches, copy the `CleanCut` folder from
`~/Library/Application Support` while CleanCut is stopped. Media paths must
match, or the destination's mounted share must be selected as a new library
root with `CLEANCUT_MEDIA_ROOTS`.

## Optional environment variables

`macos/run.sh` accepts:

- `CLEANCUT_MEDIA_ROOTS` — colon-separated library roots.
- `CLEANCUT_OUTPUT_DIR` — output/scratch location; a local SSD is recommended.
- `CLEANCUT_DATA_DIR` — persistent state and model location.
- `CLEANCUT_PORT` — web port, default `3000`.
- `CLEANCUT_HOST` — bind address, default `127.0.0.1`.
- `CLEANCUT_OLLAMA_HOST` — local Ollama endpoint, default
  `http://127.0.0.1:11434`.

Binding to `0.0.0.0` exposes the unauthenticated UI to the local network. Do
not expose this service directly to the internet.

All presets retain word-level Whisper timestamps for precise mutes, including
when a movie already contains subtitles. Whisper alignment currently runs on
the CPU because its word-timestamp operations are not supported by Metal.

## Optional actor-voice profanity replacement (beta.7)

Qwen3-TTS is a **different model and service from Ollama**. You can host it on
the same M3 Pro Mac as Ollama, but it runs through MLX-Audio on port **8765**,
not through `ollama pull` or Ollama's port 11434. No cloud speech API is used.
The companion uses its own `.venv-speech` so it cannot change CleanCut's
Whisper/NudeNet dependencies. Nothing is downloaded into Ollama's model store.
The speech runtime requires macOS 14 (Sonoma) or newer on Apple Silicon.

On the AI Mac, update the checkout and install the optional companion:

```sh
cd "$HOME/cleancut"
git pull origin codex/macos-native-v2
./macos/install-speech.sh
./macos/run-speech.sh
```

The installer downloads `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit` once
(approximately 3.1 GB of model files). Afterwards the launcher sets offline
mode and inference uses only cached weights. Leave its terminal running.
The default bind is `127.0.0.1` for CleanCut on that same Mac.

If CleanCut is running on **another Mac**, stop the speech companion with
Ctrl-C and restart it with a shared token and LAN access. Generate a token,
copy its printed value into CleanCut's **Speech service shared token** setting,
and keep this terminal running:

```sh
export CLEANCUT_SPEECH_TOKEN="$(openssl rand -hex 24)"
printf '%s\n' "$CLEANCUT_SPEECH_TOKEN"
CLEANCUT_SPEECH_BIND=0.0.0.0 ./macos/run-speech.sh
```

On restarting that terminal later, set the **same token** again; a new token
requires updating the setting. The service refuses a LAN bind without a
token. Allow the Python service through the AI Mac's firewall if macOS asks.
Do not forward port 8765 to the internet. HTTP LAN traffic is not encrypted;
use this only on a trusted home network. Client requests reject public hosts,
proxies, and redirects. Uploaded reference clips are temporary and deleted
after each request; only fitted replacement WAVs are cached with the job.

In CleanCut on the video-processing Mac:

1. Pull the same branch and restart `macos/run.sh` (or the Start command).
2. In **Settings**, leave the profanity category action set to **mute** and
   set **Audio for profanity marked “mute”** to **Replace using the actor's
   voice**. Cuts still remove scenes; other categories remain mute-only.
3. Set **Local speech service host** to the AI Mac's LAN address or `.local`
   hostname, with port **8765**. Use `http://127.0.0.1:8765` only if both
   services are on the same Mac. Set the shared token for LAN access.
4. Click **Test speech connection**, then **Save**. Perform a new scan for
   older jobs: their EDLs lack the precise replacement metadata.
5. On the scan review page, use **Preview first voice replacement** before
   rendering. A merged decision can contain several words; the button plays
   its first eligible word. Previewed WAVs are reused in the render rather
   than randomly regenerated. Render settings apply when a render is queued.

Only accepted, word-timed profanity **mutes** with an actual softened word
are eligible. Words intersecting a cut or a manually shortened decision are
skipped. The service clones a containing 3–12 second utterance, so very short
subtitle lines and explicitly marked multi-speaker lines fall back to mute.
Generated audio is silence-trimmed, level-matched, pitch-preserving
time-stretched within a conservative range, and faded at the joins. It cannot
extend into neighboring dialogue. Word mutes now have at most a 40 ms boundary
guard, and separate word mutes no longer swallow the gap between them.
Compatible video is still stream-copied. Long lists are batched to avoid
opening hundreds of WAV files at once.

If the service is unreachable, generation fails, the audio is silent, or
fitting would require excessive stretching, **the original word remains
muted**. The render log reports how many replacements succeeded. A host
failure mid-job stops further attempts instead of timing out for every word.

This is experimental dubbing, not speaker diarization or studio-quality
dialogue separation. A reference can still contain unmarked speaker changes
or music, and generated pronunciation/delivery needs listening review.
Music/effects are briefly muted with the original word too; preserving them
requires a separate dialogue-separation stage that is not implemented here.
There is no automatic ASR verification of generated words yet.

On an 18 GB Mac, avoid keeping Qwen3.5, a large Whisper model, and TTS resident
at once. After the scan, `ollama stop qwen3.5:9b` on the AI Mac can free that
model before speech generation. This does not delete its downloaded weights.
Stop the speech terminal before starting another memory-heavy scan if needed.
The companion serializes generation rather than running multiple clones in
parallel. Actual voice quality and peak memory still need testing with your
movie clips on the destination Mac.

## Cut timing fix (beta.8)

Scene cuts, mutes, subtitles, and replacement speech now share one normalized
source-to-output cut plan. Overlapping cuts count once, cut endpoints use
half-open intervals, and mutes/subtitles that overlap a cut trim to its join
instead of jumping to the start of the movie. Later cuts always remain on the
original source timeline; they are never shifted and then applied a second time.

The cut renderer no longer joins video and audio with per-segment padding,
which could accumulate timing drift at fractional-frame cut boundaries. Video
timestamps subtract the removed time directly, and audio joins on its sample
clock. Video is rounded to the source frame rate once on the final timeline.

After updating and restarting CleanCut, re-render the existing scan of the
**original movie** to repair an affected output. A new scan is not required for
this timing fix. Old scans still need rescanning to add word-level replacement
metadata if they predate beta.7. Do not use the already-edited `.clean.mp4` as
the input for the original scan's timestamps.
