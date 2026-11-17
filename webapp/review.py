"""Reading, editing and previewing an EDL from the browser.

This is the web equivalent of the CLI's interactive `review` command: the
same accept/reject/trim decisions, but driven from a page instead of a TTY.
Edits are written straight back into the EDL JSON the renderer consumes, so
what the review page shows is exactly what `clean` will apply.
"""

from __future__ import annotations

import json
import math
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from cleancut.editor_ranges import Range, normalize_cuts
from webapp.paths import job_dir

CATEGORIES = ("profanity", "drugs", "sex", "violence", "nudity")
ACTIONS = ("mute", "replace", "cut", "keep", "blur")

# A preview is for judging one decision, not for watching the film.
MAX_CLIP_SECONDS = 40.0
MAX_BLUR_SECONDS = 8
CLIP_LEAD_SECONDS = 2.0


def fmt_timestamp(seconds: float) -> str:
    whole, milliseconds = divmod(round(max(0.0, float(seconds)) * 1000), 1000)
    hours, rem = divmod(whole, 3600)
    minutes, secs = divmod(rem, 60)
    clock = f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"
    return f"{clock}.{milliseconds:03d}" if milliseconds else clock


def parse_timestamp(value: str) -> float:
    """Accept 'SS', 'MM:SS' or 'H:MM:SS' (fractional seconds allowed)."""
    parts = str(value).strip().split(":")
    if len(parts) > 3 or any(p == "" for p in parts):
        raise ValueError(f"bad timestamp: {value!r}")
    total = 0.0
    for part in parts:
        number = float(part)
        if not math.isfinite(number) or number < 0:
            raise ValueError("Timestamp components must be finite and nonnegative.")
        total = total * 60 + number
    if not math.isfinite(total):
        raise ValueError("Timestamp must be finite.")
    return total


def _cut_ranges(decisions: list[dict[str, Any]]) -> list[Range]:
    return normalize_cuts([
        Range(d["start"], d["end"]) for d in decisions
        if d.get("action") == "cut" and d.get("accepted", True)
    ])


def load_edl(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text())
    decisions = data.get("decisions", [])
    cuts = _cut_ranges(decisions)
    for index, decision in enumerate(decisions):
        decision["index"] = index
        decision["duration"] = max(0.0, decision.get("end", 0) - decision.get("start", 0))
        decision["start_label"] = fmt_timestamp(decision.get("start", 0))
        decision["end_label"] = fmt_timestamp(decision.get("end", 0))
        if decision.get("action") == "blur":
            decision["covered_by_cut"] = any(
                cut.start <= decision["start"] and decision["end"] <= cut.end for cut in cuts
            )
    return data


def save_edl(path: str | Path, data: dict[str, Any]) -> None:
    payload = {
        "video_path": data.get("video_path", ""),
        "subtitle_path": data.get("subtitle_path", ""),
        "decisions": [
            {
                "start": float(d["start"]),
                "end": float(d["end"]),
                "action": d.get("action", "mute"),
                "category": d.get("category", ""),
                "reason": d.get("reason", ""),
                "text_before": d.get("text_before", ""),
                "text_after": d.get("text_after", ""),
                "source": d.get("source", ""),
                "accepted": bool(d.get("accepted", True)),
                "word_edits": d.get("word_edits", []),
            }
            for d in data.get("decisions", [])
        ],
    }
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def summarize(data: dict[str, Any]) -> dict[str, Any]:
    """Counts the review page header shows. Only accepted decisions are
    counted under mutes/cuts -- they must match what actually renders."""
    decisions = data.get("decisions", [])
    accepted = [d for d in decisions if d.get("accepted", True)]
    by_category: dict[str, int] = {}
    by_source: dict[str, int] = {}
    for d in accepted:
        head = str(d.get("category", "")).split("+")[0] or "other"
        by_category[head] = by_category.get(head, 0) + 1
        src = str(d.get("source", "")).split("+")[0] or "other"
        by_source[src] = by_source.get(src, 0) + 1
    cuts = [d for d in accepted if d.get("action") == "cut"]
    mutes = [d for d in accepted if d.get("action") == "mute"]
    replacements = [d for d in accepted if d.get("action") == "replace"]
    blurs = [d for d in accepted if d.get("action") == "blur"]
    return {
        "total": len(decisions),
        "accepted": len(accepted),
        "rejected": len(decisions) - len(accepted),
        "cuts": len(cuts),
        "mutes": len(mutes),
        "replacements": len(replacements),
        "blurs": len(blurs),
        "nudity_review": sum(not d.get("accepted", True) and
                             str(d.get("reason", "")).startswith("[needs review;")
                             for d in decisions),
        "sex_scene_review": sum(not d.get("accepted", True) and d.get("source") == "sexual-activity"
                               for d in decisions),
        "nudity_disagreements": sum(
            "[Ollama disagrees]" in str(d.get("reason", "")) for d in decisions
        ),
        "seconds_cut": round(sum(cut.duration for cut in _cut_ranges(cuts)), 1),
        "seconds_muted": round(sum(d["end"] - d["start"] for d in mutes), 1),
        "seconds_replaced": round(sum(d["end"] - d["start"] for d in replacements), 1),
        "by_category": by_category,
        "by_source": by_source,
    }


def apply_edit(data: dict[str, Any], index: int, changes: dict[str, Any]) -> dict[str, Any]:
    decisions = data.get("decisions", [])
    if not 0 <= index < len(decisions):
        raise IndexError(f"no decision at index {index}")
    decision = dict(decisions[index])
    if changes.get("action") == "replace" and "profanity" not in str(decision.get("category", "")).split("+"):
        raise ValueError("Replace is only supported for profanity words.")
    if changes.get("action") == "blur" and not {"nudity", "sex"}.intersection(decision.get("category", "").split("+")):
        raise ValueError("Blur is available for nudity and sex decisions.")
    if "accepted" in changes:
        decision["accepted"] = bool(changes["accepted"])
    if changes.get("action") in ACTIONS:
        decision["action"] = changes["action"]
    if "start" in changes and changes["start"] not in (None, ""):
        decision["start"] = parse_timestamp(changes["start"])
    if "end" in changes and changes["end"] not in (None, ""):
        decision["end"] = parse_timestamp(changes["end"])
    if decision["end"] <= decision["start"]:
        raise ValueError("end must be after start")
    decisions[index] = decision
    return decision


def add_decision(
    data: dict[str, Any], start: float, end: float, category: str, action: str, reason: str
) -> dict[str, Any]:
    if end <= start:
        raise ValueError("end must be after start")
    if category not in CATEGORIES:
        raise ValueError(f"unknown category {category!r}")
    if action not in ACTIONS:
        raise ValueError(f"unknown action {action!r}")
    if action == "replace" and category != "profanity":
        raise ValueError("Replace is only supported for profanity words.")
    if action == "blur" and category not in {"nudity", "sex"}:
        raise ValueError("Blur is available for nudity and sex decisions.")
    decision = {
        "start": float(start), "end": float(end), "action": action, "category": category,
        "reason": f"manual: {reason}" if reason else "manual",
        "text_before": "", "text_after": "", "source": "manual", "accepted": True,
    }
    data.setdefault("decisions", []).append(decision)
    data["decisions"].sort(key=lambda d: (d["start"], d["end"]))
    return decision


def set_all_accepted(data: dict[str, Any], accepted: bool, category: str | None = None) -> int:
    changed = 0
    for decision in data.get("decisions", []):
        if category and not str(decision.get("category", "")).startswith(category):
            continue
        if bool(decision.get("accepted", True)) != accepted:
            decision["accepted"] = accepted
            changed += 1
    return changed


# --------------------------------------------------------------------------
# previews
# --------------------------------------------------------------------------

def _preview_dir(job_id: int, kind: str) -> Path:
    path = job_dir(job_id) / kind
    path.mkdir(parents=True, exist_ok=True)
    return path


def thumbnail(job_id: int, video: Path, at_seconds: float) -> Path | None:
    """One representative frame, cached per timestamp."""
    key = f"{at_seconds:.2f}".replace(".", "_")
    out = _preview_dir(job_id, "thumbs") / f"{key}.jpg"
    if out.exists():
        return out
    cmd = [
        "ffmpeg", "-nostdin", "-y", "-ss", f"{max(0.0, at_seconds):.2f}",
        "-i", str(video), "-frames:v", "1", "-vf", "scale=400:-2",
        "-q:v", "5", "-threads", "1", str(out),
    ]
    try:
        subprocess.run(cmd, capture_output=True, timeout=120, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return out if out.exists() and out.stat().st_size > 0 else None


def clip(job_id: int, video: Path, start: float, end: float) -> Path | None:
    """A short, low-bitrate excerpt around a decision.

    Re-encoded rather than stream-copied: a copy would start at the preceding
    keyframe, which can be seconds away from the moment being judged.
    """
    start = max(0.0, float(start))
    end = max(start + 0.5, float(end))
    lead = min(CLIP_LEAD_SECONDS, start)
    span = min(MAX_CLIP_SECONDS, (end - start) + lead + CLIP_LEAD_SECONDS)
    key = f"{start:.2f}_{end:.2f}".replace(".", "_")
    out = _preview_dir(job_id, "clips") / f"{key}.mp4"
    if out.exists():
        return out
    cmd = [
        "ffmpeg", "-nostdin", "-y", "-ss", f"{start - lead:.2f}", "-t", f"{span:.2f}",
        "-i", str(video),
        "-vf", "scale=640:-2", "-c:v", "libx264", "-preset", "veryfast", "-crf", "30",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "96k", "-ac", "2",
        "-movflags", "+faststart", "-threads", "2", str(out),
    ]
    try:
        subprocess.run(cmd, capture_output=True, timeout=600, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return out if out.exists() and out.stat().st_size > 0 else None


def blur_clip(job_id: int, video: Path, start: float, end: float, model: str) -> Path:
    """Render a short excerpt with the same blur logic used for the export."""
    from cleancut import cache
    from cleancut.blur import prepare_blur
    from cleancut.config import Config
    from cleancut.editor import apply_mutes_and_subs

    if not math.isfinite(start) or not math.isfinite(end) or end <= start:
        raise ValueError("Invalid blur preview range.")
    start = max(0.0, start)
    end = min(end, start + MAX_BLUR_SECONDS)
    fingerprint = video.stat()
    key = cache.config_hash(start=start, end=end, model=model, size=fingerprint.st_size,
                            mtime=fingerprint.st_mtime_ns, version=2)
    out = _preview_dir(job_id, "blur-clips") / f"{key}.mp4"
    if out.exists():
        return out
    left = round(max(0.0, start - CLIP_LEAD_SECONDS), 2)
    with tempfile.TemporaryDirectory(prefix=".preview-", dir=out.parent) as work:
        excerpt = Path(work) / "excerpt.mp4"
        subprocess.run([
            "ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", f"{left:.2f}", "-i", str(video),
            "-t", f"{end - left + 1:.6f}", "-map", "0:v:0", "-map", "0:a:0?",
            "-vf", "scale=640:-2", "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", str(excerpt),
        ], check=True, capture_output=True, timeout=120)
        config = Config(nudity_model=model)
        mask = prepare_blur(excerpt, [Range(start-left, end-left)], config, Path(work) / "blur.commands",
                            use_cache=False)
        staged = Path(work) / "preview.mp4"
        apply_mutes_and_subs(excerpt, [], None, staged, burn_subs=False, validation="quick", blur=mask)
        staged.replace(out)
    return out
