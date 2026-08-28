"""Shared helpers for LLM/VLM classifiers."""

from __future__ import annotations

import re


def strip_to_json(text: str) -> str:
    """Extract the first JSON object from a possibly chatty LLM response."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    return m.group(0) if m else text


def make_ollama_client(host: str | None):
    """Create an Ollama client, optionally with a custom host."""
    import ollama
    return ollama.Client(host=host) if host else ollama.Client()


def coerce_confidence(value) -> float:
    """LLM JSON output is schema-shaped, not schema-guaranteed — 'confidence'
    may come back as a string like "high". Treat anything non-numeric as 0.0."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def preflight_ollama(client, model: str) -> None:
    """Warm-load `model`, raising RuntimeError if the server or model is unusable.

    Without this, a down Ollama makes a scan complete with zero detections —
    indistinguishable from a clean movie.
    """
    try:
        client.generate(model=model, prompt="ok", options={"num_predict": 1})
    except Exception as e:
        raise RuntimeError(
            f"Ollama unreachable or model {model!r} unavailable: {e}. "
            f"Start it with `ollama serve` and fetch the model with `ollama pull {model}`."
        ) from e
