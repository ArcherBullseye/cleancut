import os
import time

import pytest

from cleancut import cache


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "cache")
    yield


def _make_video(tmp_path, name="movie.mp4", contents=b"abc"):
    v = tmp_path / name
    v.write_bytes(contents)
    return v


def test_save_then_load_roundtrip(tmp_path):
    v = _make_video(tmp_path)
    h = cache.config_hash(threshold=27)
    payload = {"shots": [{"start": 0, "end": 10}]}
    cache.save(v, "shots", h, payload)
    loaded = cache.load(v, "shots", h)
    assert loaded == payload


def test_load_returns_none_when_config_hash_differs(tmp_path):
    v = _make_video(tmp_path)
    cache.save(v, "shots", cache.config_hash(threshold=27), {"x": 1})
    assert cache.load(v, "shots", cache.config_hash(threshold=30)) is None


def test_load_returns_none_when_video_mtime_changes(tmp_path):
    v = _make_video(tmp_path)
    h = cache.config_hash(threshold=27)
    cache.save(v, "shots", h, {"x": 1})
    # Bump mtime
    time.sleep(0.01)
    os.utime(v, (time.time(), time.time() + 5))
    assert cache.load(v, "shots", h) is None


def test_load_returns_none_when_video_size_changes(tmp_path):
    v = _make_video(tmp_path, contents=b"abc")
    h = cache.config_hash(threshold=27)
    cache.save(v, "shots", h, {"x": 1})
    # Rewrite with different size but preserve mtime
    st = v.stat()
    v.write_bytes(b"abcdef")
    os.utime(v, (st.st_atime, st.st_mtime))
    assert cache.load(v, "shots", h) is None


def test_clear_specific_video(tmp_path):
    a = _make_video(tmp_path, "a.mp4", b"a")
    b = _make_video(tmp_path, "b.mp4", b"b")
    cache.save(a, "shots", "h1", {"x": 1})
    cache.save(b, "shots", "h1", {"x": 2})
    cache.clear(a)
    assert cache.load(a, "shots", "h1") is None
    assert cache.load(b, "shots", "h1") == {"x": 2}


def test_clear_all(tmp_path):
    a = _make_video(tmp_path, "a.mp4", b"a")
    cache.save(a, "shots", "h1", {"x": 1})
    cache.save(a, "nudenet", "h2", {"y": 2})
    assert cache.clear() == 2
