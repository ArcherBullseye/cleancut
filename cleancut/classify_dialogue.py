"""LLM-based contextual dialogue classifier.

A regex wordlist catches the word "fuck" but misses a scene that's clearly
sexual or drug-related from *context* alone. This module groups the Whisper
transcript into scene-sized chunks and asks a local LLM (via Ollama) to
classify each chunk. Chunks the LLM flags as drug/sex/violence become
EDL `cut` decisions.

Runs entirely on-device. Default model: llama3.1:8b (~5GB, ~30 tok/s on M-series).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from cleancut.constants import (
    DEFAULT_LLM_CONFIDENCE,
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
from cleancut.subtitles import Subtitle


SYSTEM_PROMPT = """You are a content classifier for a movie editing tool that removes drug and sex scenes for family viewing. Violence is HEAVILY UNDER-WEIGHTED — most "violent" content in action films stays in.

You will receive a contiguous chunk of movie dialogue with timestamps. Classify the *scene as a whole*, reading context — not just isolated words.

Output a single JSON object exactly matching this schema:
{
  "category": one of: "clean" | "profanity" | "drugs" | "sex" | "violence" | "multi",
  "should_cut": boolean,
  "confidence": number from 0.0 to 1.0,
  "reasoning": string (one short sentence)
}

Category definitions:
- "drugs"     = drug use, drug dealing, paraphernalia (pills, syringes, snorting, smoking pipes), kicking the habit, being a junkie, getting high, drug-money exchanges
- "sex"       = sexual activity, sexual coercion, sexual proposition, intimate bedroom dialogue, post-coital dialogue, prostitution, sexual seduction
- "violence"  = ONLY: torture, terrorism, weapons used against an unarmed/defenseless person, graphic injury, on-screen kill described in detail, child abuse, rape (also sex)
- "profanity" = swearing only, no drug/sex/violence content
- "clean"     = none of the above
- "multi"     = combination (e.g. drugs+sex)

Rules for should_cut:
- Profanity alone NEVER triggers should_cut=true. Mute the word; keep the scene.
- For "violence": should_cut=true ONLY if the dialogue describes torture, killing-the-helpless, or graphic non-combat violence. Fight trash-talk, combat narration, or angry threats during a fight do NOT qualify.
- For "drugs" and "sex": should_cut=true when the scene's plot purpose IS the drug/sexual content.

CRITICAL: These are NOT violence (return "clean" or "profanity"):
- "I'll kill you" / "I'll break your face" during a fight
- "Come on, you bastards" during combat
- Boxing/MMA/martial-arts commentary ("elbows to the chest", "blow to the head", "knockout")
- "Fight me" / "Bring it on" / "You're going down"
- Trash talk between fighters before/during/after a match
- Bandits or thugs threatening the hero (action-movie staple)
- Action sequences described by a sports announcer
- Mob conversations about beating someone up (without graphic detail)

These ARE violence (return "violence" or "multi"):
- A character tying up or restraining a captive
- Detailed description of how to kill or maim
- Threatening a child or non-combatant
- Mentions of rape, sexual assault, torture by name
- A character planning a real-world atrocity (mass shooting, bombing civilians)

If a scene is ambiguous between fight and violence: default to "clean" or "profanity" with should_cut=false.

Output JSON only. No prose, no markdown, no commentary."""


@dataclass
class LLMParams:
    model: str = "llama3.1:8b"
    chunk_max_seconds: float = 90.0      # max length of a scene chunk
    chunk_join_gap: float = 6.0          # gap between subtitle lines that breaks a chunk
    min_chunk_lines: int = 2             # skip chunks with fewer than this many lines
    min_confidence: float = DEFAULT_LLM_CONFIDENCE  # below this, ignore the classification
    pad_seconds: float = 1.0
    ollama_host: str | None = None       # None = default 127.0.0.1:11434


@dataclass
class DialogueChunk:
    start: float
    end: float
    lines: list[Subtitle]

    def format_for_prompt(self) -> str:
        out = []
        for s in self.lines:
            mm = int(s.start // 60)
            ss = s.start - mm * 60
            out.append(f"[{mm}:{ss:05.2f}] {s.text.strip()}")
        return "\n".join(out)


def chunk_dialogue(subs: list[Subtitle], params: LLMParams) -> list[DialogueChunk]:
    """Group consecutive subtitle lines into scene-sized chunks.

    Splits on long silence gaps (> chunk_join_gap) or when a chunk exceeds
    chunk_max_seconds. Drops tiny chunks (fewer than min_chunk_lines).
    """
    chunks: list[DialogueChunk] = []
    current: list[Subtitle] = []
    for s in subs:
        if current:
            gap = s.start - current[-1].end
            span = s.end - current[0].start
            if gap > params.chunk_join_gap or span > params.chunk_max_seconds:
                chunks.append(DialogueChunk(current[0].start, current[-1].end, list(current)))
                current = []
        current.append(s)
    if current:
        chunks.append(DialogueChunk(current[0].start, current[-1].end, list(current)))
    return [c for c in chunks if len(c.lines) >= params.min_chunk_lines]


def _classify_one(client, params: LLMParams, chunk: DialogueChunk) -> dict | None:
    """Send a single chunk to Ollama; return parsed JSON or None on failure."""
    prompt = (
        f"Scene from {chunk.start:.1f}s to {chunk.end:.1f}s "
        f"(duration {chunk.end - chunk.start:.1f}s):\n\n{chunk.format_for_prompt()}\n\n"
        "Return the JSON classification now."
    )
    try:
        resp = client.chat(
            model=params.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            format="json",
            options={"temperature": 0.0, "num_ctx": 4096},
        )
        text = resp["message"]["content"]
        return json.loads(strip_to_json(text))
    except Exception:
        return None


def classify_dialogue(
    subs: list[Subtitle],
    params: LLMParams,
    video: Path | None = None,
    use_cache: bool = True,
) -> EditDecisionList:
    """Classify every dialogue chunk; emit cut decisions for flagged scenes.

    Results cache per (video, model, params, chunk fingerprint) when `video`
    is given — at seconds per chunk, this is one of the slowest stages.
    """
    try:
        import ollama  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "LLM classifier requires the ollama python client. "
            "Install with: pip install 'cleancut[llm]'"
        ) from e

    chunks = chunk_dialogue(subs, params)
    if not chunks:
        return EditDecisionList()

    cache_hash = None
    if use_cache and video is not None:
        from cleancut import cache as _cache
        cache_hash = _cache.config_hash(
            model=params.model,
            min_confidence=params.min_confidence,
            pad_seconds=params.pad_seconds,
            chunks=[(round(c.start, 2), round(c.end, 2), len(c.lines),
                     sum(len(s.text) for s in c.lines)) for c in chunks],
        )
        hit = _cache.load(video, "llm_dialogue", cache_hash)
        if hit is not None:
            return EditDecisionList(
                decisions=[EditDecision(**d) for d in hit.get("decisions", [])]
            )

    client = make_ollama_client(params.ollama_host)

    # Warm-load the model — fails loudly if the server or model is unusable.
    preflight_ollama(client, params.model)

    edl = EditDecisionList()
    consecutive_failures = 0
    for chunk in tqdm(chunks, desc="LLM dialogue scan", unit="chunk", leave=False):
        result = _classify_one(client, params, chunk)
        if not result:
            consecutive_failures += 1
            if consecutive_failures >= MAX_CONSECUTIVE_LLM_FAILURES:
                raise RuntimeError(
                    f"{consecutive_failures} consecutive Ollama failures — aborting LLM scan."
                )
            continue
        consecutive_failures = 0
        if not result.get("should_cut"):
            continue
        confidence = coerce_confidence(result.get("confidence", 0.0))
        if confidence < params.min_confidence:
            continue
        category = str(result.get("category", "multi"))
        # Profanity alone should never cut.
        if category == "profanity":
            continue
        if category not in {"drugs", "sex", "violence", "multi"}:
            continue
        edl.add(
            EditDecision(
                start=max(0.0, chunk.start - params.pad_seconds),
                end=chunk.end + params.pad_seconds,
                action="cut",
                category=category,
                reason=f"LLM ({params.model}): {result.get('reasoning', '')[:MAX_REASON_LENGTH]}",
                source="llm-dialogue",
            )
        )

    if cache_hash is not None:
        from dataclasses import asdict

        from cleancut import cache as _cache
        _cache.save(video, "llm_dialogue", cache_hash, {
            "decisions": [asdict(d) for d in edl.decisions],
        })
    return edl
