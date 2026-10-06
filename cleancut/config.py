from __future__ import annotations

import json
import platform
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Literal

from cleancut.constants import (
    DEFAULT_SCENE_THRESHOLD,
    DEFAULT_VLM_GAPS_RADIUS,
)

Category = Literal["profanity", "drugs", "sex", "violence", "nudity"]
Action = Literal["mute", "cut", "keep"]


PRESETS = {
    "fast": {
        "visual_sample_seconds": 1.0,
        "visual_threshold": 0.55,
        "nudity_model": "fast",
        "nudity_strong_threshold": 0.85,
        "nudity_rescan_fps": 4.0,
        "snap_cuts_to_scenes": False,
        "whisper_model": "base",
        "whisper_word_timestamps": True,
        "density_enabled": False,
        "llm_enabled": False,
        "vlm_enabled": False,
        "audio_events_enabled": False,
        "encoder": "auto",
        "quality": 23,
    },
    "balanced": {
        "visual_sample_seconds": 0.5,
        "visual_threshold": 0.45,
        "nudity_model": "accurate",
        "nudity_strong_threshold": 0.80,
        "nudity_rescan_fps": 6.0,
        "snap_cuts_to_scenes": True,
        "whisper_model": "small",
        "whisper_word_timestamps": True,
        "density_enabled": True,
        "llm_enabled": False,
        "vlm_enabled": False,
        "audio_events_enabled": False,
        "encoder": "auto",
        "quality": 20,
    },
    "thorough": {
        # Default for capable hardware (e.g. M-series Mac, 32GB+ RAM).
        "visual_sample_seconds": 0.25,
        "visual_threshold": 0.35,
        "nudity_model": "accurate",
        "nudity_strong_threshold": 0.70,
        "nudity_rescan_fps": 10.0,
        "snap_cuts_to_scenes": True,
        "whisper_model": "large-v3",
        "whisper_word_timestamps": True,
        "density_enabled": True,
        "llm_enabled": True,
        "vlm_enabled": True,
        "audio_events_enabled": True,
        "encoder": "auto",
        "quality": 18,
    },
}

DEFAULT_ACTIONS: dict[str, Action] = {
    "profanity": "mute",
    "drugs": "mute",
    "sex": "mute",
    "violence": "keep",
    "nudity": "cut",
}


@dataclass
class Config:
    wordlists: dict[str, list[str]] = field(default_factory=dict)
    replacements: dict[str, str] = field(default_factory=dict)
    actions: dict[str, Action] = field(default_factory=lambda: dict(DEFAULT_ACTIONS))
    enabled_categories: set[str] = field(
        default_factory=lambda: {"profanity", "drugs", "sex", "nudity"}
    )
    # Visual sampling: examine 1 frame every N seconds.
    visual_sample_seconds: float = 1.0
    # Decode visual/scene analysis from a small proxy. Detector accuracy does
    # not benefit from feeding it 4K pixels, while decode cost and memory do.
    analysis_proxy_enabled: bool = True
    analysis_max_height: int = 720
    # First-pass NudeNet confidence threshold. Borderline hits are densely
    # rescanned before a cut is emitted, so this can favor recall.
    visual_threshold: float = 0.45
    # Accurate uses the official 640m model; fast uses bundled 320n.
    nudity_model: str = "accurate"
    nudity_strong_threshold: float = 0.80
    nudity_rescan_fps: float = 6.0
    nudity_rescan_window_seconds: float = 2.0
    nudity_min_confirmations: int = 2
    nudity_batch_size: int = 8
    nudity_coreml: bool = True
    nudity_temporal_padding_seconds: float = 0.5
    nudity_max_gap_seconds: float = 0.75
    # Deprecated compatibility settings retained for existing config files.
    visual_min_streak: int = 3
    visual_shot_hit_fraction: float = 0.5
    # Scene detection threshold for PySceneDetect ContentDetector. Lower = more cuts.
    scene_threshold: float = DEFAULT_SCENE_THRESHOLD
    # Snap dialogue cuts outward to nearest shot boundary when scenes are available.
    snap_cuts_to_scenes: bool = True
    # Pad mute/cut ranges by this many seconds on each side so cuts feel natural.
    pad_seconds: float = 0.15
    # Merge adjacent ranges closer than this.
    merge_gap_seconds: float = 0.5
    # Whisper: model name, device (None = autodetect), word-level timestamps.
    whisper_model: str = "large-v3"
    whisper_device: str | None = None
    whisper_word_timestamps: bool = True
    whisper_language: str | None = None
    # Density clustering of EDL events into "scene" cuts.
    density_enabled: bool = True
    density_window_seconds: float = 60.0
    density_min_events: int = 3
    density_min_cluster_span: float = 8.0
    # LLM-based contextual dialogue classification (via Ollama).
    llm_enabled: bool = False
    llm_model: str = "qwen3.5:9b" if platform.system() == "Darwin" else "llama3.1:8b"
    llm_host: str | None = None
    llm_min_confidence: float = 0.6
    # VLM-based visual scene classification (via Ollama).
    vlm_enabled: bool = False
    vlm_model: str = "qwen3.5:9b" if platform.system() == "Darwin" else "llava:7b"
    vlm_mode: str = "silent+gaps"
    vlm_stride: int = 1
    vlm_min_confidence: float = 0.55
    vlm_cut_intimate: bool = False
    vlm_gaps_radius: float = DEFAULT_VLM_GAPS_RADIUS
    # Audio event detection (HuggingFace AST on AudioSet).
    audio_events_enabled: bool = False
    audio_events_model: str = "MIT/ast-finetuned-audioset-10-10-0.4593"
    audio_events_threshold: float = 0.45
    audio_events_clip_seconds: float = 8.0
    audio_events_skip_violence: bool = True
    # Cross-signal corroboration is reserved for the broad VLM classifier.
    # Temporally confirmed NudeNet detections do not require audible dialogue.
    require_visual_corroboration: bool = True
    corroboration_radius_seconds: float = 5.0
    # Encoder choice for the final render.
    # "videotoolbox" = Apple Silicon hardware H.264 (fast)
    # "hevc_videotoolbox" = Apple Silicon hardware HEVC Main10 (HDR-safe)
    # "libx264" = software (best quality, slower)
    # "auto" = videotoolbox on macOS, libx264 elsewhere
    encoder: str = "auto"
    # Quality target. For libx264: CRF (lower = better). For videotoolbox: q (higher = better).
    quality: int = 20
    # none | quick (metadata/duration) | full (decode every output frame)
    render_validation: str = "quick"
    # Speech is served by a local MLX companion, not Ollama's text API.
    profanity_audio: str = "mute"  # mute | replace (experimental, mute fallback)
    speech_host: str = "http://127.0.0.1:8765"
    speech_model: str = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit"
    speech_token: str = ""
    speech_timeout: float = 120.0

    @classmethod
    def load_defaults(cls) -> Config:
        return cls(
            wordlists=_load_packaged_json("wordlists.json"),
            replacements=_load_packaged_json("replacements.json"),
        )

    def apply_preset(self, name: str) -> None:
        if name not in PRESETS:
            raise ValueError(f"Unknown preset {name!r}. Options: {list(PRESETS)}")
        for k, v in PRESETS[name].items():
            setattr(self, k, v)

    def resolved_encoder(self, video_path: Path | None = None) -> str:
        if self.encoder == "auto":
            if platform.system() == "Darwin":
                if video_path is not None:
                    try:
                        from cleancut.probe import probe_streams, video_stream

                        source = video_stream(probe_streams(video_path))
                        if source and source.is_hdr:
                            return "hevc_videotoolbox"
                    except Exception:
                        pass
                return "videotoolbox"
            return "libx264"
        return self.encoder

    def override_wordlists(self, path: Path | None) -> None:
        if path:
            self.wordlists = json.loads(Path(path).read_text())

    def override_replacements(self, path: Path | None) -> None:
        if path:
            self.replacements = json.loads(Path(path).read_text())


def _load_packaged_json(name: str) -> dict:
    pkg = resources.files("cleancut.data")
    return json.loads(pkg.joinpath(name).read_text())
