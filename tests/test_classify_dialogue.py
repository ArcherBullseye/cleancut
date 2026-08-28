from cleancut.classify_dialogue import LLMParams, chunk_dialogue
from cleancut.subtitles import Subtitle


def _sub(idx, start, end, text):
    return Subtitle(index=idx, start=start, end=end, text=text)


def test_chunk_splits_on_long_gap():
    subs = [
        _sub(1, 0, 2, "Hello"),
        _sub(2, 3, 5, "How are you"),
        _sub(3, 30, 32, "Much later"),
        _sub(4, 33, 35, "Continuing"),
    ]
    chunks = chunk_dialogue(subs, LLMParams(chunk_join_gap=10, min_chunk_lines=2))
    assert len(chunks) == 2
    assert chunks[0].start == 0
    assert chunks[0].end == 5
    assert chunks[1].start == 30


def test_chunk_splits_on_max_span():
    subs = [_sub(i, i * 2.0, i * 2.0 + 1.5, f"line {i}") for i in range(60)]
    # 60 lines spaced 2s apart — total span 120s. With chunk_max_seconds=30, expect 4 chunks.
    chunks = chunk_dialogue(subs, LLMParams(chunk_max_seconds=30, chunk_join_gap=10, min_chunk_lines=2))
    assert len(chunks) >= 3


def test_drops_singleton_chunks():
    subs = [_sub(1, 0, 1, "Lonely"), _sub(2, 60, 61, "Also lonely")]
    chunks = chunk_dialogue(subs, LLMParams(chunk_join_gap=5, min_chunk_lines=2))
    assert chunks == []
