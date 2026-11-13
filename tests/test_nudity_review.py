from __future__ import annotations

import json
import shutil
import subprocess
from unittest.mock import Mock

import pytest

from cleancut.config import Config
from cleancut.edl import EditDecision, EditDecisionList
from cleancut.edl_ops import cuts_for_review
from cleancut.nudity_review import review_nudity
from cleancut.visual import NudityEvidence
from webapp.review import summarize


@pytest.fixture
def video(tmp_path, monkeypatch):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("Requires FFmpeg")
    pytest.importorskip("cv2")
    path = tmp_path / "source.mkv"
    subprocess.run([
        "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=black:s=32x32:r=10:d=4",
        "-vf", "drawbox=c=white:t=fill:enable='eq(n,1)+eq(n,25)'",
        "-c:v", "ffv1", str(path),
    ], check=True)
    monkeypatch.setattr("cleancut.cache.CACHE_DIR", tmp_path / "cache")
    return path


def _reply(verdict):
    return {"message": {"content": json.dumps({"verdict": verdict})}}


def test_verification_uses_the_exact_frame_and_new_hits_stay_unselected(video, monkeypatch):
    import cv2
    import numpy as np

    client = Mock()
    images_per_call = []

    def classify(**kwargs):
        images = kwargs["messages"][-1]["images"]
        frames = [cv2.imdecode(np.frombuffer(img, dtype=np.uint8), cv2.IMREAD_COLOR) for img in images]
        images_per_call.append(frames)
        if len(frames) > 1:
            return _reply("absent" if len(images_per_call) == 1 else "visible")
        return _reply("visible" if frames[0].mean() > 200 else "absent")

    client.chat.side_effect = classify
    factory = Mock(return_value=client)
    monkeypatch.setattr("cleancut.nudity_review.make_ollama_client", factory)
    decisions = [
        EditDecision(0, 0.6, "cut", "nudity", source="visual"),
        EditDecision(0.9, 1.4, "cut", "nudity", source="visual", accepted=False),
    ]
    evidence = [NudityEvidence(1, 0.1, [4, 4, 8, 8]), NudityEvidence(10, 1.0, None)]
    config = Config(vlm_model="qwen3.5:9b")
    checked, weak, extra = review_nudity(video, video, decisions, evidence, config)

    assert [len(frames) for frames in images_per_call] == [4, 3, 1, 1, 1, 1]
    exact, crop, before, after = images_per_call[0]
    assert exact.mean() > 200 and crop.mean() > 200 and crop.shape[:2] == (12, 12)
    assert before.mean() < 10 and after.mean() < 10
    assert checked.accepted and (checked.start, checked.end) == (0, 0.6)
    assert "[Ollama disagrees]" in checked.reason
    assert not weak.accepted and "[Ollama agrees]" in weak.reason
    assert not extra.accepted and extra.source == "ollama-nudity"
    assert (extra.start, extra.end) == (2.0, 3.0)
    edl = EditDecisionList(decisions=[checked, weak, extra])
    assert edl.by_action("cut") == [checked]
    assert cuts_for_review(edl) == [checked, weak, extra]
    summary = summarize({"decisions": [vars(d) for d in edl]})
    assert summary["nudity_disagreements"] == 1 and summary["nudity_review"] == 2
    assert review_nudity(video, video, decisions, evidence, config) == list(edl)
    assert factory.call_count == 1


def test_independent_pass_resumes_after_a_model_error(video, monkeypatch):
    monkeypatch.setattr("cleancut.nudity_review.CHECKPOINT_SAMPLES", 1)
    client = Mock()
    factory = Mock(return_value=client)
    monkeypatch.setattr("cleancut.nudity_review.make_ollama_client", factory)
    client.chat.side_effect = [_reply("visible"), _reply("absent"), ValueError("failed"), ValueError("failed")]
    config = Config()
    with pytest.raises(RuntimeError, match="Ollama vision model"):
        review_nudity(video, video, [], [], config)

    client.chat.reset_mock()
    client.chat.side_effect = [_reply("visible"), _reply("absent")]
    early, late = review_nudity(video, video, [], [], config)
    assert (early.start, early.end) == (0.0, 1.0)
    assert (late.start, late.end) == (2.0, 3.0)
    assert client.chat.call_count == 2
