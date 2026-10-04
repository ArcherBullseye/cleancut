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

The balanced/thorough presets retain word-level Whisper timestamps for precise
mutes, which currently runs Whisper on the CPU. The fast preset disables word
timestamps and can use Metal (MPS), trading some mute precision for speed.
