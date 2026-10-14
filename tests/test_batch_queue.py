"""Folder selection and durable, atomic multi-video queueing."""
import json
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest

from webapp import jobs, library, settings
from webapp.app import app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "_SETTINGS_PATH", tmp_path / "settings.json")
    monkeypatch.setattr(jobs, "DB_PATH", tmp_path / "jobs.db")
    monkeypatch.setattr(jobs, "ensure_dirs", lambda: None)
    monkeypatch.setattr(jobs, "job_dir", lambda i: tmp_path / "jobs" / str(i))
    monkeypatch.setattr(library, "allowed_roots", lambda: [tmp_path / "media"])
    (tmp_path / "media").mkdir()
    jobs.init_db()
    return app.test_client()


def video(tmp_path, name):
    path = tmp_path / "media" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def test_batch_order_settings_deduplication_and_restart(client, tmp_path):
    a, b = video(tmp_path, "a.mp4"), video(tmp_path, "b.mkv")
    body = {"paths": [str(b), str(a), str(b)], "preset": "fast", "auto_render": True,
            "actions": {"profanity": "replace"}, "word_end_padding_ms": 300}
    response = client.post("/api/scan", json=body)
    assert response.status_code == 200, response.json
    assert response.json["queued_count"] == 2
    records = [jobs.get_job(item["job_id"]) for item in response.json["jobs"]]
    assert [r["video_path"] for r in records] == [str(b), str(a)]
    assert all(r["status"] == "queued" and r["preset"] == "fast" for r in records)
    assert all(json.loads(r["options"])["auto_render"] for r in records)
    assert settings.load()["word_end_padding_ms"] == 300
    settings.save({"actions": {"profanity": "mute"}})
    assert all(json.loads(r["options"])["actions"]["profanity"] == "replace" for r in records)
    jobs.init_db()
    assert jobs._next_queued()["id"] == records[0]["id"]
    retry = client.post("/api/scan", json=body).json
    assert retry["queued_count"] == 0
    assert all(r["existing"] for r in retry["jobs"])
    assert len(jobs.list_jobs()) == 2


@pytest.mark.parametrize("bad", ["missing.mp4", "notes.txt", "../outside.mp4"])
def test_invalid_file_rejects_entire_batch(client, tmp_path, bad):
    a = video(tmp_path, "a.mp4")
    video(tmp_path, "notes.txt")
    (tmp_path / "outside.mp4").touch()
    result = client.post("/api/scan", json={"paths": [str(a), str(a.parent / bad)]})
    assert result.status_code in {400, 403}
    assert jobs.list_jobs() == []
    assert not settings._SETTINGS_PATH.exists()


@pytest.mark.parametrize("body", [[], {"paths": []}, {"paths": "x"}, {"paths": [None]},
                                  {"paths": ["x"] * 501}])
def test_malformed_batch_is_rejected(client, body):
    assert client.post("/api/scan", json=body).status_code == 400
    assert not jobs.list_jobs()


def test_folder_recursive_safe_deduplicated_and_skips_outputs(client, tmp_path):
    a, b = video(tmp_path, "a.mp4"), video(tmp_path, "season/b.mkv")
    for name in ["a.clean.mp4", "notes.txt", ".hidden/a.mp4", "@Recycle/a.mp4"]:
        video(tmp_path, name)
    outside = tmp_path / "private.mp4"
    outside.touch()
    (a.parent / "outside.mp4").symlink_to(outside)
    (a.parent / "alias.mp4").symlink_to(a)
    (a.parent / "season" / "loop").symlink_to(a.parent, target_is_directory=True)
    response = client.post("/api/folder-videos", json={"path": str(a.parent)} )
    assert response.status_code == 200
    assert [f["path"] for f in response.json["files"]] == [str(a), str(b)]
    direct = client.post("/api/folder-videos", json={"path": str(a.parent), "recursive": False})
    assert [f["path"] for f in direct.json["files"]] == [str(a)]
    assert not jobs.list_jobs()  # selecting is not submitting
    assert client.post("/api/folder-videos", json={"path": str(tmp_path)}).status_code == 403


def test_folder_limit_is_explicit_not_truncated(client, tmp_path):
    a = video(tmp_path, "a.mp4")
    video(tmp_path, "b.mp4")
    with patch.object(library, "MAX_BATCH_FILES", 1):
        result = client.post("/api/folder-videos", json={"path": str(a.parent)})
    assert result.status_code == 400 and "smaller folder" in result.json["error"]
    assert not jobs.list_jobs()


def test_empty_folder_returns_empty_selection(client, tmp_path):
    response = client.post("/api/folder-videos", json={"path": str(tmp_path / "media")})
    assert response.json == {"files": []}


def test_concurrent_batches_reuse_active_jobs(client, tmp_path):
    videos = [video(tmp_path, "a.mp4"), video(tmp_path, "b.mp4")]
    def submit(_):
        return jobs.create_scan_jobs(videos, preset="fast", options={})
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(submit, range(2)))
    assert [r["job_id"] for r in results[0]] == [r["job_id"] for r in results[1]]
    assert len(jobs.list_jobs()) == 2


def test_storage_failure_rolls_back_batch(client, tmp_path):
    videos = [video(tmp_path, "a.mp4"), video(tmp_path, "b.mp4")]
    with patch.object(jobs, "job_dir", side_effect=[tmp_path / "first", OSError("disk full")]), \
         pytest.raises(OSError):
        jobs.create_scan_jobs(videos, preset="fast", options={})
    assert jobs.list_jobs() == []


def test_library_has_batch_controls(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b'"add-folder"' in response.data and b'"selected-files"' in response.data


def test_all_active_jobs_are_visible_beyond_history_limit(client, tmp_path):
    videos = [video(tmp_path, f"{i}.mp4") for i in range(120)]
    jobs.create_scan_jobs(videos, preset="fast", options={})
    assert len(client.get("/api/jobs").json["jobs"]) == 120


def test_same_named_videos_get_distinct_output_paths(client, tmp_path):
    videos = [video(tmp_path, f"season{i}/episode.mkv") for i in (1, 2)]
    records = jobs.create_scan_jobs(videos, preset="fast",
        options={"output_dir": str(tmp_path / "output"), "output_location": "folder"})
    for record in records:
        jobs.update_job(record["job_id"], status="done", edl_path=str(tmp_path / "edl.json"))
    first, second = [jobs.get_job(jobs.queue_render(record["job_id"])) for record in records]
    assert first["output_path"] != second["output_path"]
    again = jobs.get_job(jobs.queue_render(records[1]["job_id"]))
    assert again["output_path"] == second["output_path"]
