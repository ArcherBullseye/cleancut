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
    assert len(jobs.list_jobs()) == 1
    batch = jobs.get_job(response.json["batch_id"])
    assert batch["kind"] == "batch" and batch["file_count"] == 2
    assert retry["batch_id"] == batch["id"]


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


def make_batch(client, tmp_path, **options):
    paths = [str(video(tmp_path, f"batch/{i}.mp4")) for i in range(2)]
    response = client.post("/api/scan", json={"paths": paths, **options})
    assert response.status_code == 200
    return response.json["batch_id"], [item["job_id"] for item in response.json["jobs"]]


def test_parent_status_tracks_children_and_never_runs_as_subprocess(client, tmp_path):
    parent, children = make_batch(client, tmp_path)
    assert [job["id"] for job in client.get("/api/jobs").json["jobs"]] == [parent]
    jobs.update_job(children[0], status="running")
    assert jobs.get_job(parent)["status"] == "running"
    assert jobs._next_queued()["id"] == children[1]
    for child in children:
        jobs.update_job(child, status="done")
    assert jobs.get_job(parent)["status"] == "done"
    assert jobs.get_job(parent)["progress"] == 100
    assert jobs._next_queued() is None
    jobs.init_db()
    assert jobs.get_job(parent)["status"] == "done"


def test_render_is_nested_and_batch_waits_for_auto_render(client, tmp_path):
    parent, children = make_batch(client, tmp_path, auto_render=True)
    settings.save({"output_dir": str(tmp_path / "out")})
    for child in children:
        jobs.update_job(child, status="done", edl_path=str(tmp_path / "edl.json"))
    assert jobs.get_job(parent)["status"] == "queued"
    with patch.object(jobs, "OUTPUT_DIR", tmp_path / "out"):
        renders = [jobs.queue_render(child) for child in children]
    assert all(jobs.get_job(render)["batch_id"] == parent for render in renders)
    assert len(client.get("/api/jobs").json["jobs"]) == 1
    for render in renders:
        jobs.update_job(render, status="done")
    assert jobs.get_job(parent)["status"] == "done"


def test_bulk_delete_filters_protect_active_and_keep_videos(client, tmp_path):
    parent, children = make_batch(client, tmp_path)
    original = jobs.get_job(children[0])["video_path"]
    from pathlib import Path
    cleaned = tmp_path / "movie.clean.mp4"
    cleaned.write_bytes(b"cleaned")
    for child in children:
        jobs.update_job(child, status="done")
    active = jobs.create_job("scan", str(video(tmp_path, "active.mp4")))
    assert [j["id"] for j in client.get("/api/jobs?state=done").json["jobs"]] == [parent]
    result = client.post("/api/jobs/delete", json={"ids": [parent, active]}).json
    assert result == {"deleted": [parent], "skipped": [active]}
    assert jobs.get_job(parent) is None
    assert all(jobs.get_job(child) is None for child in children)
    assert Path(original).exists() and cleaned.read_bytes() == b"cleaned"
    assert all(not jobs.job_dir(child).exists() for child in children)


def test_bulk_delete_scan_with_active_render_is_skipped(client, tmp_path):
    scan = jobs.create_job("scan", str(video(tmp_path, "movie.mp4")))
    jobs.update_job(scan, status="done")
    render = jobs.create_job("render", jobs.get_job(scan)["video_path"], parent_id=scan)
    assert client.post("/api/jobs/delete", json={"ids": [scan]}).json["skipped"] == [scan]
    assert jobs.get_job(render)


def test_cancel_batch_cancels_every_queued_file(client, tmp_path):
    parent, children = make_batch(client, tmp_path)
    assert client.post(f"/api/job/{parent}/cancel").json["ok"]
    assert all(jobs.get_job(child)["status"] == "canceled" for child in children)
    assert jobs.get_job(parent)["status"] == "canceled"


@pytest.mark.parametrize("body", [{}, {"ids": []}, {"ids": [True]}, {"ids": "all"}])
def test_bulk_delete_requires_explicit_ids(client, body):
    assert client.post("/api/jobs/delete", json=body).status_code == 400
