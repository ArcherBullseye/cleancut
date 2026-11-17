from __future__ import annotations

from importlib.metadata import version
from unittest.mock import Mock

import numpy as np
import pytest

from cleancut.config import Config
from cleancut.nudity_model import bundled_model
from cleancut.visual import _CompactNudeNetSession, _open_detector


def test_compaction_preserves_original_thresholds_boxes_and_nms(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    implementation = pytest.importorskip("nudenet.nudenet")
    if version("nudenet") != "3.4.2":
        pytest.skip("Tests the supported NudeNet 3.4.2 output format")

    outputs = np.zeros((4, 22, 30), dtype=np.float32)
    outputs[:, :4] = np.array([155, 150, 100, 80])[None, :, None]
    # Zero or one surviving anchors exercise the shape constraint. Tied
    # scores on overlapping boxes exercise the original suppression order.
    outputs[1, 4, 25] = 0.9
    outputs[2, 8, 19] = outputs[2, 5, 4] = 0.9
    outputs[2, 4, 23] = 0.249
    outputs[2, 4, 24] = np.nan
    outputs[2, 4, 25] = 0.2
    outputs[2, 4, 26] = 0.251
    outputs[2, 4, 27] = 0.75
    outputs[2, 0, 26] = 0
    outputs[2, 0, 27] = 300
    outputs[3, 4, 0] = 0.2
    session = Mock()
    session.run.return_value = [outputs]
    compact, = _CompactNudeNetSession(session).run(None, {"images": object()})
    assert compact.shape == (4, 22, 6)
    args = (0, 32, 1.0, 1.0, 640, 608, 320, 320)
    for original, filtered in zip(outputs, compact, strict=True):
        assert implementation._postprocess([filtered], *args) == implementation._postprocess([original], *args)


def test_output_adapter_leaves_unsupported_layouts_unchanged():
    session = Mock()
    for outputs in ([np.zeros((1, 6, 100))], [np.zeros((1, 22, 2))], [np.zeros(9), np.zeros(9)]):
        session.run.return_value = outputs
        assert _CompactNudeNetSession(session).run() is outputs


def test_bundled_detector_returns_identical_results_with_compaction(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pytest.importorskip("nudenet.nudenet")
    if version("nudenet") != "3.4.2":
        pytest.skip("The optimization only applies to NudeNet 3.4.2")

    _, detector = _open_detector(Config(nudity_coreml=False), bundled_model())
    assert isinstance(detector.onnx_session, _CompactNudeNetSession)
    rng = np.random.default_rng(52)
    images = [rng.integers(0, 256, (360, 640, 3), dtype=np.uint8) for _ in range(3)]
    fast_batch = detector.detect_batch(images, batch_size=3)
    fast_single = detector.detect(images[0])
    detector.onnx_session = detector.onnx_session._session
    assert fast_batch == detector.detect_batch(images, batch_size=3)
    assert fast_single == detector.detect(images[0])
