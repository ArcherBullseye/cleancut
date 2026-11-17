from __future__ import annotations

from threading import Event
from unittest.mock import Mock

import pytest

from cleancut.config import Config
from cleancut.scenes import Shot
from cleancut.sexual_activity import _shot_verdicts


def _stub_decoder(monkeypatch, frame_count, encode):
    pytest.importorskip("cv2")
    cap = Mock()
    monkeypatch.setattr("cleancut.sexual_activity.probe_video_timing", lambda _video: (0, frame_count / 10, None))
    monkeypatch.setattr("cleancut.sexual_activity._open_capture", Mock(return_value=cap))
    monkeypatch.setattr("cleancut.sexual_activity.iter_frame_positions",
                        lambda *_args: ((index, index / 10) for index in range(frame_count)))
    monkeypatch.setattr("cleancut.sexual_activity._frame_image", lambda _cap, timestamp: round(timestamp * 10))
    monkeypatch.setattr("cleancut.sexual_activity._image_bytes", lambda _cv2, frame: encode(frame))
    monkeypatch.setattr("cleancut.sexual_activity.make_ollama_client", Mock())
    return cap


def test_model_request_overlaps_decoding_the_next_shot(tmp_path, monkeypatch):
    second_shot_ready = Event()

    def encode(frame):
        if frame >= 10:
            second_shot_ready.set()
        return bytes([frame])

    cap = _stub_decoder(monkeypatch, 20, encode)
    requests = []

    def judge(_client, _config, images):
        requests.append(images)
        if len(requests) == 1:
            assert second_shot_ready.wait(2), "decoder waited for the first model response"
        return "absent"

    monkeypatch.setattr("cleancut.sexual_activity._judge", judge)
    verdicts = _shot_verdicts(tmp_path / "video.mkv", [Shot(0, 1), Shot(1, 2)], Config(),
                             previous=[], checkpoint=Mock())

    assert verdicts == ["absent", "absent"]
    assert requests == [[b"\x00", b"\x08"], [b"\x0a", b"\x12"]]
    cap.release.assert_called_once()


def test_prefetched_windows_after_an_active_verdict_do_not_reach_the_model(tmp_path, monkeypatch):
    cap = _stub_decoder(monkeypatch, 66, lambda frame: bytes([frame]))
    judge = Mock(side_effect=["active", "absent"])
    monkeypatch.setattr("cleancut.sexual_activity._judge", judge)
    verdicts = _shot_verdicts(tmp_path / "video.mkv", [Shot(0, 6), Shot(6, 6.6)], Config(),
                             previous=[], checkpoint=Mock())

    assert verdicts == ["active", "absent"]
    assert [call.args[2][0] for call in judge.call_args_list] == [b"\x00", b"\x3c"]
    cap.release.assert_called_once()
