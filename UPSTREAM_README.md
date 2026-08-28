# cleancut

[![CI](https://github.com/monahand1023/cleancut/actions/workflows/ci.yml/badge.svg)](https://github.com/monahand1023/cleancut/actions/workflows/ci.yml) [![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/) [![License: MIT](https://img.shields.io/github/license/monahand1023/cleancut)](LICENSE)

> Auto-edit movies for content — mutes profanity, cuts explicit scenes, softens subtitles. Runs entirely on-device.

cleancut detects profanity, drug references, sex, and nudity using a layered stack of local signals, then mutes audio, cuts scenes, and rewrites subtitles to softer text. The output is a cleaned `.mp4` with a toggleable softened subtitle track.

**No cloud APIs.** All AI runs locally — Whisper for speech-to-text, Ollama-hosted LLMs for dialogue context, NudeNet + LLaVA for vision, HuggingFace AST for audio events.

## How it works

```mermaid
flowchart TB
    V["Input video"] --> A["Audio track"]
    V --> F["Video frames"]
    A --> W["Whisper STT<br/>transcript + timing"]
    W --> LLM["Ollama LLM<br/>profanity · drugs · dialogue context"]
    A --> AST["HuggingFace AST<br/>audio events"]
    F --> VIS["NudeNet + LLaVA<br/>sex / nudity detection"]
    LLM --> EDL["Edit Decision List<br/>(auditable — every cut is logged)"]
    AST --> EDL
    VIS --> EDL
    EDL --> FF["ffmpeg<br/>mute · cut · soften subtitles"]
    FF --> OUT["Cleaned .mp4"]
```

---

## Contents

- [How it works](#how-it-works)
- [Signal stack](#signal-stack)
- [Requirements](#requirements)
- [Install](#install)
- [Quickstart](#quickstart)
- [Subcommands](#subcommands)
- [Presets](#presets)
- [Configuration](#configuration)
- [Categories and actions](#categories-and-actions)
- [EDL format](#edl-format)
- [Caching](#caching)
- [Troubleshooting](#troubleshooting)
- [Legal](#legal)
- [License](#license)

---

## How it works

cleancut runs a layered detection pipeline on a video file and produces an **Edit Decision List (EDL)** — a JSON file describing every proposed cut or mute. The pipeline:

```mermaid
flowchart TD
    video(["video.mp4"]) --> probe["probe — detect audio/subtitle tracks, language"]
    video --> transcribe["transcribe — Whisper STT if no .srt"]
    video --> subscan["subtitle scan — wordlist match + context gating"]
    subscan --> density["density — cluster nearby hits into content scenes"]
    video --> llm["LLM dialogue — Ollama/Llama 3.1 classifies ambiguous scenes"]
    video --> nudenet["NudeNet — frame-level explicit nudity detection"]
    video --> vlm["VLM scene — LLaVA classifies intimate framing / paraphernalia"]
    video --> audio["Audio events — HF AST classifies moans, screams, etc."]
    probe --> merge["All signals merge"]
    transcribe --> merge
    density --> merge
    llm --> merge
    nudenet --> merge
    vlm --> merge
    audio --> merge
    merge --> edl["EDL"]
    edl --> review["optional review"]
    review --> render(["render"])
```

Each signal runs independently. Results merge into one EDL, then density clustering and snap-to-shot post-processing dedupe and align the cuts to clean shot boundaries. The report shows which signal triggered each cut, so every decision is auditable.

---

## Signal stack

| # | Signal | Catches | Cost | Tool |
|---|--------|---------|------|------|
| 1 | **Wordlist match** | Explicit profanity / drugs / sex words | free | regex |
| 2 | **Context gating** | Drops ambiguous weak hits (e.g. "blow" alone) without strong neighbors | free | regex |
| 3 | **Density clustering** | Clusters of any wordlist hits → likely a "content scene" | free | Python |
| 4 | **LLM dialogue** | Scenes wordlists miss ("as good as your ex", drug pushing) | free local | Ollama + Llama 3.1 8B |
| 5 | **NudeNet** | Explicit nudity (technical / narrow definition) | free local | NudeNet |
| 6 | **VLM scene** | Intimate framing without explicit nudity, drug paraphernalia | free local | Ollama + LLaVA 7B |
| 7 | **Audio events** | Moans, screams, gunshots — semantic sound classification | free local | HuggingFace AST |

Visual signals (NudeNet, VLM) require **corroboration** from at least one non-visual signal before a cut fires — this eliminates false positives from action scenes, medical shots, etc.

---

## Requirements

**System:**
- macOS (Apple Silicon recommended) or Linux
- Python 3.10+
- `ffmpeg` with `libass` for subtitle burn-in (optional; soft-sub fallback always works)

**Hardware (for full `thorough` preset):**
- 16 GB RAM minimum; 32 GB recommended
- Apple M-series or CUDA GPU for acceptable Whisper and LLaVA speed
- ~15 GB free disk for model weights (Whisper large-v3 ~3 GB, Llama 3.1 8B ~5 GB, LLaVA 7B ~5 GB)

**Lighter configs (no GPU, 8 GB RAM):** use `--preset fast` or `--preset balanced` — skips LLM, VLM, and audio event detection.

---

## Install

Install system dependencies first:

```bash
brew install ffmpeg ollama
```

Pull the Ollama models you need (only required for `thorough` preset):

```bash
ollama serve &
ollama pull llama3.1:8b   # LLM dialogue (~5 GB)
ollama pull llava:7b      # VLM visual (~5 GB)
```

Install cleancut:

```bash
git clone https://github.com/monahand1023/cleancut.git
cd cleancut
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[full]"   # everything: Whisper + NudeNet + PySceneDetect + Ollama + AST
```

**Lighter installs** (pick what you need):

```bash
pip install -e .                        # subtitle-only — no STT, no visual, no LLM
pip install -e ".[whisper]"             # + Whisper transcription
pip install -e ".[visual,scenes]"       # + NudeNet + PySceneDetect
pip install -e ".[llm]"                 # + Ollama client
pip install -e ".[audio]"               # + HuggingFace AST audio events
pip install -e ".[full]"                # everything
```

Verify install:

```bash
cleancut --help
```

---

## Quickstart

### Full workflow (scan → review → render)

```bash
# 1. Inspect the file — see what tracks are available
cleancut inspect /path/to/movie.mp4

# 2. Scan — produces an EDL + report, no video output yet
cleancut scan /path/to/movie.mp4 \
  --preset thorough \
  --save-transcript /path/to/movie.whisper.srt \
  -o /path/to/movie.edl.json

# 3. Review interactively — approve/reject/trim each cut
cleancut review /path/to/movie.edl.json \
  --subs /path/to/movie.whisper.srt

# 4. Render the cleaned video
cleancut clean /path/to/movie.mp4 \
  --edl /path/to/movie.edl.json \
  --subs /path/to/movie.whisper.srt \
  -o /path/to/movie.clean.mp4

# 5. Add a manual cut if the review missed something
cleancut add-cut /path/to/movie.edl.json \
  --start 58:32 --end 1:00:08 \
  --category sex --reason "intimate scene"
```

### One-shot (scan + render, no review)

```bash
cleancut clean /path/to/movie.mp4 \
  --preset thorough \
  --save-transcript /path/to/movie.whisper.srt \
  -o /path/to/movie.clean.mp4
```

---

## Subcommands

### `cleancut inspect FILE`

Shows audio tracks, subtitle tracks (text vs image-based), sidecar `.srt` files found nearby, and a "plan" of what `clean` would do given current flags. Useful for understanding the file before committing to a long scan.

```
Track 0  audio    eng  aac    stereo
Track 1  audio    spa  aac    stereo
Track 2  subtitle eng  (text) subrip
→ will use: audio track 0 (eng), subtitle track 2
```

---

### `cleancut scan FILE`

Runs the full signal stack (or the subset enabled by your preset/flags) and writes:

- `<file>.edl.json` — every decision (accepted and rejected)
- `<file>.edl.report.txt` — human-readable summary grouped by category and signal source

Options:

| Flag | Description |
|------|-------------|
| `--preset fast\|balanced\|thorough` | Signal configuration (see [Presets](#presets)) |
| `--subs FILE` | Use this `.srt` instead of running Whisper |
| `--save-transcript FILE` | Cache Whisper output for future runs |
| `--no-visual` | Skip NudeNet + VLM |
| `--no-llm` | Skip Ollama dialogue classification |
| `--no-audio` | Skip HuggingFace audio events |
| `--language CODE` | Prefer this language track (`eng`, `spa`, …) |
| `-o FILE` | Write EDL to this path (default: alongside video) |

---

### `cleancut review EDL`

Interactive walkthrough of every accepted cut. For each one, cleancut extracts a representative frame, shows surrounding dialogue context, and prompts:

| Key | Action |
|-----|--------|
| `y` | Keep this cut |
| `n` | Reject (sets `accepted=false` in the EDL) |
| `t START END` | Trim the cut range (`MM:SS` or `H:MM:SS`) |
| `o` | Open the extracted frame in your default viewer |
| `s` | Skip without changing |
| `q` | Save and quit |

By default, pure-violence cuts are hidden (fight scenes preserved). Pass `--include-violence` to review those too.

```bash
cleancut review movie.edl.json --subs movie.whisper.srt --include-violence
```

---

### `cleancut add-cut EDL`

Manually insert a cut into an existing EDL:

```bash
cleancut add-cut movie.edl.json \
  --start 1:05:22 --end 1:06:44 \
  --category sex \
  --reason "bedroom scene missed by VLM" \
  --snap    # extend to nearest shot boundaries
```

---

### `cleancut clean FILE`

Apply an EDL and render the cleaned video. If `--edl` is omitted, runs a fresh scan first.

```bash
cleancut clean movie.mp4 \
  --edl movie.edl.json \
  --subs movie.whisper.srt \
  -o movie.clean.mp4
```

The subtitle track is written as a **soft (non-burned) track** in the output container (`mov_text` for .mp4/.m4v/.mov, `srt` for .mkv and others) — toggle it on/off in VLC, Infuse, Plex, or any standard player. If ffmpeg is compiled with `--enable-libass`, subtitles are burned in instead.

---

## Presets

```bash
cleancut scan FILE --preset {fast|balanced|thorough}
```

| Preset | Whisper model | Visual sample rate | Density | LLM | VLM | Audio events | Notes |
|--------|---------------|--------------------|---------|-----|-----|--------------|-------|
| `fast` | `base` | 2.0 s | off | off | off | off | Quick first pass |
| `balanced` | `small` + words | 1.0 s | on | off | off | off | Good default for fast machines |
| `thorough` | `large-v3` + words | 0.5 s | on | on | on | on | **Default.** Best quality |

`thorough` is the default. On an M-series Mac with 32 GB+ RAM it takes roughly:

| Stage | Time (90 min film) |
|-------|--------------------|
| Whisper large-v3 | 20–35 min |
| NudeNet | 10–20 min |
| LLaVA VLM | 15–30 min |
| AST audio events | 5–10 min |
| LLM dialogue | 2–5 min |

All stages cache — re-runs that reuse cached results take ~30 seconds.

---

## Configuration

### Wordlists and replacements

Wordlists live in `cleancut/data/wordlists.json`. Each entry can be a plain string or an object:

```json
{
  "profanity": [
    "fuck",
    { "pattern": "crap", "strength": "weak" }
  ]
}
```

**Weak** patterns only fire when a **strong** hit (any category) is within ±30 seconds. This eliminates false positives like "blow" matching "blow to the head" in a boxing scene.

**Replacements** live in `cleancut/data/replacements.json` and map explicit subtitle text to softer alternatives shown in the cleaned output:

```json
{
  "fuck": "frick",
  "shit": "shoot"
}
```

Override with your own files:

```bash
cleancut clean FILE \
  --wordlists my_words.json \
  --replacements my_replacements.json
```

### Per-category action overrides

```bash
# Cut profanity instead of muting
cleancut clean FILE --action profanity=cut

# Enable violence cuts (off by default)
cleancut clean FILE --enable-category violence

# Disable nudity cuts
cleancut clean FILE --disable-category nudity
```

### Audio track selection

```bash
# Force a specific audio track (0-indexed)
cleancut clean FILE --audio-track 1

# Prefer Spanish audio/subtitles
cleancut clean FILE --language spa
```

---

## Categories and actions

| Category | Default action | Notes |
|----------|----------------|-------|
| `profanity` | `mute` + soften | Audio muted, subtitle text replaced |
| `drugs` | `mute` + soften | Same |
| `sex` | `mute` + soften | Visual matches also trigger `cut` |
| `nudity` | `cut` | Visual-only (NudeNet, VLM) |
| `violence` | `keep` | Off by default — fight scenes preserved |

Actions: `mute` (silence audio, keep video), `cut` (remove segment entirely), `keep` (no-op).

---

## EDL format

The `.edl.json` file is human-readable JSON. You can edit it directly before rendering.

```json
{
  "video_path": "/path/to/movie.mp4",
  "subtitle_path": "/path/to/movie.whisper.srt",
  "decisions": [
    {
      "start": 312.4,
      "end": 315.1,
      "action": "mute",
      "category": "profanity",
      "reason": "wordlist: fuck (strong)",
      "text_before": "What the fuck are you doing?",
      "text_after": "What the frick are you doing?",
      "source": "subtitle+whisper",
      "accepted": true
    },
    {
      "start": 3742.0,
      "end": 3798.5,
      "action": "cut",
      "category": "nudity",
      "reason": "NudeNet: EXPOSED_BREAST_F (0.94) corroborated by AST: moaning",
      "text_before": "",
      "text_after": "",
      "source": "visual+audio",
      "accepted": true
    }
  ]
}
```

Fields:

| Field | Type | Description |
|-------|------|-------------|
| `start` / `end` | float | Seconds from start of video |
| `action` | string | `"mute"`, `"cut"`, or `"keep"` |
| `category` | string | `"profanity"`, `"drugs"`, `"sex"`, `"nudity"`, `"violence"` |
| `reason` | string | Human-readable explanation with signal + confidence |
| `text_before` | string | Original subtitle text (empty for visual-only cuts) |
| `text_after` | string | Softened replacement text |
| `source` | string | Which signals fired, joined by `+` |
| `accepted` | bool | Set to `false` during review to suppress the cut |

---

## Caching

All heavy computations cache to `~/.cache/cleancut/` keyed by video file (mtime + size) and config. Cached results are automatically reused on re-runs:

| Stage | Cache type | Time saved |
|-------|-----------|-----------|
| Whisper transcript | Auto-cached (also: `--save-transcript FILE` once, `--subs FILE` on re-runs) | 20–35 min |
| PySceneDetect shot boundaries | Auto-cached | 5–10 min |
| NudeNet visual scan | Auto-cached | 10–30 min |
| LLM dialogue classification | Auto-cached | 2–5 min |
| LLaVA VLM scene scan | Auto-cached | 15–30 min |
| AST audio events | Auto-cached | 5–10 min |

Cache is invalidated automatically when the video file changes (mtime or size), and entries for different configs coexist — toggling presets doesn't evict anything. If you saved a transcript with `--save-transcript`, the `.words.json` written alongside it is read back on `--subs` re-runs, so word-precise mutes survive across runs.

---

## Troubleshooting

**Whisper takes forever / runs out of memory**
Use a smaller model via `--preset balanced` (small) or `--preset fast` (base), or pre-generate the transcript once and reuse it with `--subs`.

**LLaVA / LLM not running**
cleancut now fails loudly if Ollama is unreachable or a model is missing (the LLM/VLM stage is skipped with a visible warning rather than silently reporting nothing). Ensure `ollama serve` is running (`ps aux | grep ollama`) and pull the models if missing:
```bash
ollama pull llama3.1:8b
ollama pull llava:7b
```

**"libass not found" warning**
Subtitles fall back to a soft (non-burned) track automatically. To enable burn-in, install ffmpeg with libass:
```bash
brew install ffmpeg   # Homebrew's ffmpeg includes libass
```

**False positives (cuts you didn't want)**
Run `cleancut review EDL` and reject them with `n`. Then check if a wordlist pattern is too broad — weak patterns in `wordlists.json` should cover most ambiguous cases.

**False negatives (scenes not caught)**
Use `cleancut add-cut` to manually insert a cut, then check whether the signal that should have caught it was enabled in your preset.

---

## Legal

The [Family Movie Act of 2005](https://en.wikipedia.org/wiki/Family_Movie_Act) (US) permits private home performance of filtered authorized copies of motion pictures. cleancut is intended for **personal use on media you own**. Do not redistribute cleaned output.

---

## License

MIT — see [LICENSE](LICENSE) for details.
