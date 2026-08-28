"""Vision-language model (VLM) scene classifier.

Closes the gap that NudeNet leaves: scenes that are visually intimate / drug-
related / violent but don't show explicit body parts the narrow NudeNet
classifier was trained on. Also catches silent shots that have no dialogue
for the LLM dialogue classifier to read (e.g. the bra close-up at the start
of the Ong Bak dealer scene).

Runs locally via Ollama. Default model: llava:7b (~5GB, ~2-4s per frame on M-series).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from cleancut.constants import (
    DEFAULT_VLM_CONFIDENCE,
    DEFAULT_VLM_GAPS_RADIUS,
    DEFAULT_VLM_MIN_SHOT_DURATION,
    MAX_CONSECUTIVE_LLM_FAILURES,
    MAX_REASON_LENGTH,
)
from cleancut.edl import EditDecision, EditDecisionList
from cleancut.llm_utils import (
    coerce_confidence,
    make_ollama_client,
    preflight_ollama,
    strip_to_json,
)
from cleancut.scenes import Shot
from cleancut.subtitles import Subtitle


SYSTEM_PROMPT = """You are a content classifier for a movie editing tool that removes drug, sex, and violent scenes for family viewing. You will be shown a single frame from a movie.

Return a single JSON object exactly matching this schema:
{
  "intimate":   boolean,   // bedroom scene, kissing, undressing, bra/underwear visible, two people in bed
  "explicit":   boolean,   // visible nudity (breasts, genitals, buttocks)
  "drug_use":   boolean,   // syringes, snorting, pills being consumed, smoking pipes, paraphernalia
  "violence":   boolean,   // active violence with blood, weapons being used to harm (NOT martial-arts fighting, NOT sports combat)
  "confidence": number,    // 0.0-1.0 — overall confidence in your classification
  "description": string    // one short sentence: what the frame depicts
}

Rules:
- A shirtless man, a fight scene, or a martial-arts kick is NOT violence
- Two people kissing or in bed = intimate=true (even if clothed)
- A bra visible in a non-sexual context (e.g. changing) is still intimate=true
- Dark/black frames or simple establishing shots = all false
- Output JSON only. No prose."""


@dataclass
class VLMParams:
    model: str = "llava:7b"
    # How to pick which shots to classify.
    # "all" - every shot (slow, thorough)
    # "stride" - every Nth shot
    # "silent" - only shots with no dialogue overlapping (catches visual-only scenes)
    # "gaps" - shots adjacent to already-flagged ranges (extends existing scenes)
    mode: str = "silent+gaps"
    stride: int = 1
    # Adjacent-shot search radius for "gaps" mode (in seconds).
    gaps_radius_seconds: float = DEFAULT_VLM_GAPS_RADIUS
    # Skip shots shorter than this — they're usually transitions.
    min_shot_duration: float = DEFAULT_VLM_MIN_SHOT_DURATION
    # Confidence threshold.
    min_confidence: float = DEFAULT_VLM_CONFIDENCE
    # Categories to consider a cut. "intimate" is borderline; off by default,
    # but turned on by `--vlm-cut-intimate`.
    cut_on: tuple[str, ...] = ("explicit", "drug_use", "violence")
    ollama_host: str | None = None


def _ranges_with_dialogue(subs: list[Subtitle]) -> list[tuple[float, float]]:
    return [(s.start, s.end) for s in subs]


def _has_dialogue_overlap(shot: Shot, dialogue_ranges: list[tuple[float, float]]) -> bool:
    for ds, de in dialogue_ranges:
        if ds < shot.end and de > shot.start:
            return True
    return False


def _shot_adjacent_to_flagged(shot: Shot, flagged: list[tuple[float, float]], radius: float) -> bool:
    for fs, fe in flagged:
        if shot.start < fe + radius and shot.end > fs - radius:
            return True
    return False


def select_shots(
    shots: list[Shot],
    subs: list[Subtitle],
    existing_edl: EditDecisionList,
    params: VLMParams,
) -> list[Shot]:
    """Decide which shots get sent to the VLM, per the selected mode."""
    shots = [s for s in shots if s.duration >= params.min_shot_duration]
    if params.mode == "all":
        return shots
    if params.mode == "stride":
        return shots[::max(1, params.stride)]

    # Compose mode flags.
    modes = set(params.mode.split("+"))
    dialogue = _ranges_with_dialogue(subs) if "silent" in modes else []
    flagged = [
        (d.start, d.end) for d in existing_edl.decisions if d.action in ("mute", "cut")
    ] if "gaps" in modes else []

    out: list[Shot] = []
    for s in shots:
        keep = False
        if "silent" in modes and not _has_dialogue_overlap(s, dialogue):
            keep = True
        if "gaps" in modes and _shot_adjacent_to_flagged(s, flagged, params.gaps_radius_seconds):
            keep = True
        if "all" in modes:
            keep = True
        if keep:
            out.append(s)
    return out


def _extract_frame(video: Path, t: float, tmp_dir: Path) -> Path | None:
    """Extract a single frame at time `t` (seconds) to a JPEG. Returns path or None."""
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found on PATH.")
    from cleancut.edl_ops import fmt_ffmpeg_timestamp

    out = tmp_dir / f"frame_{int(t * 1000)}.jpg"
    timestamp = fmt_ffmpeg_timestamp(t)
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-v", "error",
                "-ss", timestamp,
                "-i", str(video),
                "-frames:v", "1", "-q:v", "3",
                str(out),
            ],
            check=True,
        )
        return out if out.exists() else None
    except subprocess.CalledProcessError:
        return None


def _classify_frame(client, params: VLMParams, frame_path: Path) -> dict | None:
    try:
        resp = client.chat(
            model=params.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": "Classify this movie frame. Return JSON only.",
                    "images": [str(frame_path)],
                },
            ],
            format="json",
            options={"temperature": 0.0},
        )
        text = resp["message"]["content"]
        return json.loads(strip_to_json(text))
    except Exception:
        return None


def _flagged_categories(result: dict, params: VLMParams) -> list[str]:
    cats: list[str] = []
    if not result:
        return cats
    if coerce_confidence(result.get("confidence", 0.0)) < params.min_confidence:
        return cats
    mapping = {
        "explicit": "nudity",
        "intimate": "sex",
        "drug_use": "drugs",
        "violence": "violence",
    }
    for key in params.cut_on:
        if result.get(key) is True:
            cats.append(mapping.get(key, key))
    return cats


def scan_with_vlm(
    video: Path,
    shots: list[Shot],
    subs: list[Subtitle],
    existing_edl: EditDecisionList,
    params: VLMParams,
    use_cache: bool = True,
) -> EditDecisionList:
    """Run VLM over a selected subset of shots; emit cuts for flagged shots."""
    try:
        import ollama  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "VLM classifier requires the ollama python client. "
            "Install with: pip install 'cleancut[llm]'"
        ) from e

    targets = select_shots(shots, subs, existing_edl, params)
    if not targets:
        return EditDecisionList()

    # The target list already encodes mode/stride/gaps selection, so it (plus
    # scoring params) fully keys the result. LLaVA runs seconds per frame —
    # cache like the other detectors.
    cache_hash = None
    if use_cache:
        from cleancut import cache as _cache
        cache_hash = _cache.config_hash(
            model=params.model,
            min_confidence=params.min_confidence,
            cut_on=sorted(params.cut_on),
            targets=[(round(s.start, 2), round(s.end, 2)) for s in targets],
        )
        hit = _cache.load(video, "vlm", cache_hash)
        if hit is not None:
            return EditDecisionList(
                decisions=[EditDecision(**d) for d in hit.get("decisions", [])]
            )

    client = make_ollama_client(params.ollama_host)
    # Warm-load the model — fails loudly if the server or model is unusable.
    preflight_ollama(client, params.model)

    edl = EditDecisionList()
    consecutive_failures = 0
    with tempfile.TemporaryDirectory(prefix="cleancut_vlm_") as tmp:
        tmp_dir = Path(tmp)
        for shot in tqdm(targets, desc=f"VLM scan ({params.mode})", unit="shot", leave=False):
            t = shot.start + shot.duration / 2.0
            frame = _extract_frame(video, t, tmp_dir)
            if frame is None:
                continue
            result = _classify_frame(client, params, frame)
            if result is None:
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_LLM_FAILURES:
                    raise RuntimeError(
                        f"{consecutive_failures} consecutive Ollama failures — aborting VLM scan."
                    )
                continue
            consecutive_failures = 0
            cats = _flagged_categories(result, params)
            if not cats:
                continue
            category = "+".join(cats)
            desc = (result or {}).get("description", "")[:MAX_REASON_LENGTH]
            edl.add(
                EditDecision(
                    start=shot.start,
                    end=shot.end,
                    action="cut",
                    category=category,
                    reason=f"VLM ({params.model}) {','.join(cats)}: {desc}",
                    source="vlm",
                )
            )

    if cache_hash is not None:
        from dataclasses import asdict

        from cleancut import cache as _cache
        _cache.save(video, "vlm", cache_hash, {
            "decisions": [asdict(d) for d in edl.decisions],
        })
    return edl
