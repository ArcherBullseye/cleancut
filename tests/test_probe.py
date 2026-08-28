"""Tests for probe.py — sidecar discovery, language preference, track selection."""

import pytest

from cleancut.probe import (
    Stream,
    audio_streams,
    find_sidecar_subtitle,
    pick_audio_track,
    pick_embedded_subtitle,
    subtitle_streams,
)


def _audio(idx, lang, codec="aac", channels=2, title=""):
    return Stream(index=idx, codec_name=codec, codec_type="audio",
                  language=lang, title=title, channels=channels)


def _sub(idx, lang, codec="dvd_subtitle", title=""):
    return Stream(index=idx, codec_name=codec, codec_type="subtitle",
                  language=lang, title=title)


def _video(idx=0):
    return Stream(index=idx, codec_name="h264", codec_type="video")


def test_audio_streams_filter():
    streams = [_video(), _audio(1, "eng"), _audio(2, "spa"), _sub(3, "eng")]
    assert len(audio_streams(streams)) == 2


def test_subtitle_streams_filter():
    streams = [_video(), _audio(1, "eng"), _sub(2, "eng"), _sub(3, "spa")]
    assert len(subtitle_streams(streams)) == 2


def test_pick_audio_track_prefers_english_by_default():
    streams = [_video(), _audio(1, "tha"), _audio(2, "eng")]
    picked = pick_audio_track(streams, requested=None, prefer_language="eng")
    assert picked.index == 2


def test_pick_audio_track_falls_back_to_first():
    streams = [_video(), _audio(1, "tha"), _audio(2, "fra")]
    picked = pick_audio_track(streams, requested=None, prefer_language="eng")
    assert picked.index == 1


def test_pick_audio_track_explicit_request():
    streams = [_video(), _audio(1, "tha"), _audio(2, "eng")]
    picked = pick_audio_track(streams, requested=0, prefer_language="eng")
    assert picked.language == "tha"


def test_pick_audio_track_invalid_request_raises():
    streams = [_video(), _audio(1, "tha")]
    with pytest.raises(ValueError):
        pick_audio_track(streams, requested=5)


def test_pick_audio_track_returns_none_when_no_audio():
    streams = [_video()]
    assert pick_audio_track(streams, requested=None) is None


def test_pick_embedded_subtitle_skips_image_codecs():
    streams = [_sub(1, "eng", codec="dvd_subtitle"), _sub(2, "eng", codec="hdmv_pgs_subtitle")]
    assert pick_embedded_subtitle(streams) is None


def test_pick_embedded_subtitle_prefers_text():
    streams = [_sub(1, "eng", codec="dvd_subtitle"), _sub(2, "eng", codec="mov_text")]
    picked = pick_embedded_subtitle(streams)
    assert picked is not None
    assert picked.codec_name == "mov_text"


def test_pick_embedded_subtitle_language_preference():
    streams = [_sub(1, "fra", codec="srt"), _sub(2, "eng", codec="srt")]
    picked = pick_embedded_subtitle(streams, prefer_language="eng")
    assert picked.language == "eng"


def test_find_sidecar_bare_name(tmp_path):
    video = tmp_path / "Movie.mp4"
    video.write_bytes(b"x")
    (tmp_path / "Movie.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nHi\n")
    found = find_sidecar_subtitle(video)
    assert found.name == "Movie.srt"


def test_find_sidecar_prefers_english_suffix(tmp_path):
    video = tmp_path / "Movie.mp4"
    video.write_bytes(b"x")
    (tmp_path / "Movie.fr.srt").write_text("x")
    (tmp_path / "Movie.en.srt").write_text("x")
    (tmp_path / "Movie.es.srt").write_text("x")
    found = find_sidecar_subtitle(video, prefer_language="eng")
    assert found.name == "Movie.en.srt"


def test_find_sidecar_penalizes_sdh(tmp_path):
    video = tmp_path / "Movie.mp4"
    video.write_bytes(b"x")
    (tmp_path / "Movie.en.srt").write_text("x")
    (tmp_path / "Movie.en.sdh.srt").write_text("x")
    found = find_sidecar_subtitle(video, prefer_language="eng")
    assert "sdh" not in found.name


def test_find_sidecar_searches_subs_subfolder(tmp_path):
    video = tmp_path / "Movie.mp4"
    video.write_bytes(b"x")
    subs_dir = tmp_path / "Subs"
    subs_dir.mkdir()
    (subs_dir / "english.srt").write_text("x")
    found = find_sidecar_subtitle(video, prefer_language="eng")
    assert found.parent == subs_dir


def test_find_sidecar_returns_none_when_no_srt(tmp_path):
    video = tmp_path / "Movie.mp4"
    video.write_bytes(b"x")
    assert find_sidecar_subtitle(video) is None
