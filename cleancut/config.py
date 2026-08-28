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
    DEFAULT_VISUAL_THRESHOLD,
)

Category = Literal["profanity", "drugs", "sex", "violence", "nudity"]
Action = Literal["mute", "cut", "keep"]


PRESETS = {
    "fast": {
        "visual_sample_seconds": 2.0,
        "visual_min_streak": 3,
        "visual_shot_hit_fraction": 0.6,
        "visual_threshold": 0.75,
        "snap_cuts_to_scenes": False,
        "whisper_model": "base",
        "whisper_word_timestamps": False,
        "density_enabled": False,
        "llm_enabled": False,
        "vlm_enabled": False,
        "audio_events_enabled": False,
        "encoder": "auto",
        "quality": 23,
    },
    "balanced": {
        "visual_sample_seconds": 1.0,
        "visual_min_streak": 3,
        "visual_shot_hit_fraction": 0.5,
        "visual_threshold": 0.7,
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
        "visual_sample_seconds": 0.5,
        "visual_min_streak": 3,
        "visual_shot_hit_fraction": 0.45,
        "visual_threshold": 0.65,
        "snap_cuts_to_scenes": True,
        "whisper_model": "large-v3",
        "whisper_word_timestamps": True,
        "density_enabled": True,
        "llm_enabled": True,
        "vlm_enabled": True,
        "audio_events_enabled": True,
        "encoder": "libx264",
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
    # NudeNet confidence threshold for explicit-class detections.
    # 0.7 chosen after testing — 0.55 fired on shirtless men in action films.
    visual_threshold: float = DEFAULT_VISUAL_THRESHOLD
    # Streak mode: require this many consecutive flagged samples to cut.
    visual_min_streak: int = 3
    # Shot-aware mode: fraction of sampled frames within a shot that must hit.
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
    llm_model: str = "llama3.1:8b"
    llm_host: str | None = None
    llm_min_confidence: float = 0.6
    # VLM-based visual scene classification (via Ollama).
    vlm_enabled: bool = False
    vlm_model: str = "llava:7b"
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
    # Cross-signal corroboration: require visual-only cuts (NudeNet, VLM) to have
    # a dialogue/audio event within ±N seconds. False positives on visual-only
    # detectors are common (shirtless men, cigarettes, explosions) — corroboration
    # demands a second signal before committing the cut.
    require_visual_corroboration: bool = True
    corroboration_radius_seconds: float = 5.0
    # Encoder choice for the final render.
    # "videotoolbox" = Apple Silicon hardware H.264 (fast)
    # "libx264" = software (best quality, slower)
    # "auto" = videotoolbox on macOS, libx264 elsewhere
    encoder: str = "auto"
    # Quality target. For libx264: CRF (lower = better). For videotoolbox: q (higher = better).
    quality: int = 20

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

    def resolved_encoder(self) -> str:
        if self.encoder == "auto":
            return "videotoolbox" if platform.system() == "Darwin" else "libx264"
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
