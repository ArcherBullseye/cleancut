from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import patch

import pytest

from cleancut import nudity_model


class Response:
    def __init__(self, data: bytes):
        self.data = data
        self.offset = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def read(self, size: int) -> bytes:
        chunk = self.data[self.offset:self.offset + size]
        self.offset += len(chunk)
        return chunk


def test_accurate_model_download_is_checksum_verified_and_atomic(tmp_path):
    data = b"valid local onnx model"
    digest = hashlib.sha256(data).hexdigest()
    with patch("cleancut.nudity_model.model_dir", return_value=tmp_path), \
         patch.object(nudity_model, "NUDENET_640M_SIZE", len(data)), \
         patch.object(nudity_model, "NUDENET_640M_SHA256", digest), \
         patch("urllib.request.urlopen", return_value=Response(data)) as opened:
        path = nudity_model.ensure_accurate_model()
    assert path.read_bytes() == data
    assert not path.with_suffix(".onnx.partial").exists()
    request = opened.call_args.args[0]
    assert request.headers["Accept"] == "application/octet-stream"


def test_bad_model_download_is_deleted(tmp_path):
    data = b"not the expected model"
    with patch("cleancut.nudity_model.model_dir", return_value=tmp_path), \
         patch.object(nudity_model, "NUDENET_640M_SIZE", len(data)), \
         patch.object(nudity_model, "NUDENET_640M_SHA256", "0" * 64), \
         patch("urllib.request.urlopen", return_value=Response(data)), \
         pytest.raises(RuntimeError, match="checksum"):
        nudity_model.ensure_accurate_model()
    assert not list(tmp_path.glob("*.partial"))
    assert not (tmp_path / nudity_model.NUDENET_640M_FILENAME).exists()


def test_accurate_model_failure_falls_back_to_bundled(tmp_path):
    fallback = nudity_model.NudityModel("NudeNet-320n", Path("/tmp/320n.onnx"), 320)
    with patch("cleancut.nudity_model.ensure_accurate_model", side_effect=OSError("offline")), \
         patch("cleancut.nudity_model.bundled_model", return_value=fallback):
        assert nudity_model.resolve_nudity_model("accurate") == fallback
