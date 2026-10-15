"""All network/NAS effects isolated to temporary fixtures; never contact live Macs."""
import base64
import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from cleancut import media_validation, probe
from cleancut.edl import EditDecision, EditDecisionList
from webapp import integrations as it
from webapp import jobs, library, settings
from webapp.app import app


@pytest.fixture
def env(tmp_path, monkeypatch):
    media = tmp_path / "media"
    media.mkdir()
    source = media / "movie.mkv"
    source.write_bytes(b"original-video")
    monkeypatch.setattr(it, "CONFIG_PATH", tmp_path / "integrations.json")
    monkeypatch.setattr(settings, "_SETTINGS_PATH", tmp_path / "settings.json")
    monkeypatch.setattr(jobs, "DB_PATH", tmp_path / "jobs.db")
    monkeypatch.setattr(jobs, "ensure_dirs", lambda: None)
    monkeypatch.setattr(jobs, "job_dir", lambda i: tmp_path / "jobs" / str(i))
    monkeypatch.setattr(library, "allowed_roots", lambda: [media])
    jobs.init_db()
    it.init_db()
    it.save_config({"enabled": True, "services": {name: {
        "url": f"http://127.0.0.1:{7878 if name == 'radarr' else 8989}", "api_key": "secret-key",
        "local_root": str(media), "server_root": "/nas"} for name in it.SERVICES}})
    calls = []

    def api(service, resource, body=None):
        calls.append((resource, body))
        responses = {
            "movie/1": {"tags": [5], "movieFile": {"id": 2}},
            "series/1": {"tags": [5]}, "tag": [{"id": 5, "label": "cleancut"}],
            "moviefile/2": {"movieId": 1, "path": "/nas/movie.mkv"},
            "episodefile/2": {"seriesId": 1, "path": "/nas/movie.mkv"},
            "command": {"id": 12}, "command/12": {"status": "completed"},
            "system/status": {"version": "test"}, "rootfolder": [{"path": "/nas"}],
        }
        return responses[resource]

    monkeypatch.setattr(it, "arr", api)
    return SimpleNamespace(source=source, media=media, calls=calls, api=api, client=app.test_client())


def payload(name="radarr", **extra):
    owner, file = ("movie", "movieFile") if name == "radarr" else ("series", "episodeFile")
    return {"eventType": "Download", owner: {"id": 1}, file: {"id": 2, "path": "/evil/ignored"}, **extra}


def auth():
    raw = "cleancut:" + it.config()["token"]
    return {"Authorization": "Basic " + base64.b64encode(raw.encode()).decode()}


def render_ready(env, monkeypatch):
    result = it.receive("radarr", payload())
    scan = jobs.get_job(result["job_id"])
    it.preflight(scan)
    edl = jobs.edl_path_for(scan["id"])
    EditDecisionList(decisions=[EditDecision(2, 4, "cut", "nudity"),
                                EditDecision(3, 5, "cut", "nudity")]).to_json(edl)
    jobs._finish_ok(scan["id"], scan)
    render_id = jobs.queue_render(scan["id"], overrides={"render_validation": "none"})
    render = jobs.get_job(render_id)
    jobs.update_job(render_id, status=jobs.RUNNING)
    it.preflight(render)
    stage = jobs.render_stage_path(render)
    stage.write_bytes(b"cleaned-video")
    monkeypatch.setattr(probe, "probe_duration", lambda p: 10)
    monkeypatch.setattr(probe, "probe_streams", lambda p: [
        SimpleNamespace(codec_type="video", width=1920, height=1080),
        SimpleNamespace(codec_type="audio")])
    validations = []
    monkeypatch.setattr(media_validation, "validate_media", lambda p, **kw: validations.append((p, kw)))
    return result["event_id"], render, stage, validations


@pytest.mark.parametrize("name", ["radarr", "sonarr"])
def test_auth_payloads_dedupe_and_snapshots(env, name):
    client = env.client
    url = f"/api/integrations/{name}/webhook"
    assert client.post(url, json=payload(name)).status_code == 401
    assert client.post(url, headers=auth(), json={"eventType": "Test"}).json["test"]
    assert client.post(url, headers=auth(), json={"eventType": "Grab"}).json["ignored"]
    first = client.post(url, headers=auth(), json=payload(name)).json
    second = client.post(url, headers=auth(), json=payload(name, isUpgrade=True)).json
    assert first["job_id"] == second["job_id"] and second["existing"]
    scan = jobs.get_job(first["job_id"])
    opts = json.loads(scan["options"])
    assert scan["video_path"] == str(env.source)
    assert opts["auto_render"] is False and opts["integration_id"] == first["event_id"]
    assert "secret-key" not in scan["options"] and "speech_token" not in opts
    assert len(jobs.list_jobs()) == 1
    jobs.init_db()
    it.init_db()
    assert it.receive(name, payload(name))["existing"]


def test_concurrent_delivery_creates_one_job(env):
    with ThreadPoolExecutor(max_workers=5) as pool:
        results = list(pool.map(lambda _: it.receive("radarr", payload()), range(10)))
    assert len({r["job_id"] for r in results}) == 1


def test_configs_secrets_local_admin_and_csrf(env):
    client = env.client
    data = client.get("/api/integrations").json
    assert "secret-key" not in json.dumps(data) and it.config()["token"] not in json.dumps(data)
    assert it.CONFIG_PATH.stat().st_mode & 0o777 == 0o600
    assert client.get("/api/integrations", environ_base={"REMOTE_ADDR": "192.168.1.2"}).status_code == 403
    assert client.get("/api/integrations", headers={"Host": "attacker.example"}).status_code == 403
    assert client.post("/api/integrations", json={}, headers={"Origin": "https://attacker.example"}).status_code == 403
    assert client.post("/api/integrations", data="{}").status_code == 415
    assert client.post("/api/integrations", json={"services": {"radarr": {"api_key": ""}}}).status_code == 200
    assert it.config()["services"]["radarr"]["api_key"] == "secret-key"
    old = auth()
    client.post("/api/integrations", json={"rotate_token": True})
    assert client.post("/api/integrations/radarr/webhook", headers=old, json={"eventType": "Test"}).status_code == 401
    assert client.post("/api/integrations/token", json={}).headers["Cache-Control"] == "no-store"


@pytest.mark.parametrize("url", ["https://example.com", "http://8.8.8.8", "http://169.254.169.254",
                                 "file:///tmp/key", "http://user:pw@127.0.0.1", "http://127.0.0.1/?key=x"])
def test_nonlocal_urls_rejected(env, url):
    with pytest.raises(ValueError):
        it.local_url(url)


@pytest.mark.parametrize("raw", ["/nas/../movie.mkv", "/nas2/movie.mkv", "movie.mkv", "/nas/missing.mkv", "/nas/movie.avi"])
def test_path_mapping_rejects_escapes_and_missing_files(env, raw):
    with pytest.raises(ValueError):
        it.map_path(it.config()["services"]["radarr"], raw)


def test_symlinks_rejected(env, tmp_path):
    (env.media / "linked.mkv").symlink_to(env.source)
    with pytest.raises(ValueError):
        it.map_path(it.config()["services"]["radarr"], "/nas/linked.mkv")


def test_no_optin_no_job(env, monkeypatch):
    monkeypatch.setattr(it, "arr", lambda service, resource, body=None:
                        [] if resource == "tag" else env.api(service, resource, body))
    with pytest.raises(ValueError, match="opt-in"):
        it.receive("radarr", payload())
    assert not jobs.list_jobs()


def test_disabled_and_malformed(env):
    for bad in [None, [], {}, {"eventType": "Download", "movie": []}, payload(movie={"id": True})]:
        result = env.client.post("/api/integrations/radarr/webhook", headers=auth(), json=bad)
        assert result.status_code in {200, 409, 413}
        assert not jobs.list_jobs()
    it.save_config({"enabled": False})
    with pytest.raises(ValueError, match="disabled"):
        it.receive("radarr", payload())


def test_successful_replace_validated_and_rescanned(env, monkeypatch):
    event_id, render, stage, validations = render_ready(env, monkeypatch)
    assert render["output_path"] == str(env.source)
    assert stage.suffix == ".mkv"
    assert json.loads(render["options"])["render_validation"] == "full"
    jobs._finish_ok(render["id"], render)
    assert env.source.read_bytes() == b"cleaned-video"
    assert jobs.get_job(render["id"])["status"] == jobs.DONE
    event = it.get_event(event_id)
    assert event["state"] == "complete" and not Path(event["backup"]).exists()
    assert validations[0][1] == {"mode": "full", "expected_duration": 7, "expected_dimensions": (1920, 1080)}
    assert ("command", {"name": "RescanMovie", "movieId": 1}) in env.calls
    assert it.receive("radarr", payload())["existing"]  # rescan doesn't loop
    assert env.client.get(f"/api/job/{render['id']}").json["integration_state"] == "complete"
    assert env.client.get(f"/api/job/{render['id']}/download").mimetype == "video/x-matroska"
    assert env.client.post(f"/api/job/{render['parent_id']}/render", json={}).status_code == 409
    from webapp.app import _video_for
    assert _video_for(render) is None


def test_upgrade_before_render_refused(env, monkeypatch):
    event_id = it.receive("radarr", payload())["event_id"]
    env.source.write_bytes(b"new-upgrade")
    with pytest.raises(ValueError, match="changed"):
        it.preflight(jobs.get_job(it.get_event(event_id)["job_id"]))
    assert env.source.read_bytes() == b"new-upgrade"


def test_upgrade_during_render_refused(env, monkeypatch):
    _, render, stage, _ = render_ready(env, monkeypatch)
    env.source.write_bytes(b"new-upgrade")
    with pytest.raises(ValueError, match="changed"):
        it.publish(render, stage)
    assert env.source.read_bytes() == b"new-upgrade"


def test_verification_failure_preserves_original(env, monkeypatch):
    _, render, stage, _ = render_ready(env, monkeypatch)
    def fail(*a, **k):
        raise RuntimeError("bad frames")
    monkeypatch.setattr(media_validation, "validate_media", fail)
    jobs._finish_ok(render["id"], render)
    assert env.source.read_bytes() == b"original-video"
    assert jobs.get_job(render["id"])["status"] == jobs.FAILED
    assert stage.exists()


def test_failed_rescan_backup_retained_recovery_finishes(env, monkeypatch):
    event_id, render, _stage, _ = render_ready(env, monkeypatch)
    def fail(event):
        raise ValueError("rescan unavailable")
    original = it._rescan
    monkeypatch.setattr(it, "_rescan", fail)
    jobs._finish_ok(render["id"], render)
    event = it.get_event(event_id)
    assert event["state"] == "published" and Path(event["backup"]).read_bytes() == b"original-video"
    assert env.source.read_bytes() == b"cleaned-video"
    assert env.client.delete(f"/api/job/{render['id']}").status_code == 409
    assert env.client.post("/api/jobs/delete", json={"ids": [event["job_id"]]}).json["skipped"] == [event["job_id"]]
    monkeypatch.setattr(it, "_rescan", original)
    it.recover(event_id)
    assert it.get_event(event_id)["state"] == "complete"
    assert not Path(event["backup"]).exists()
    assert jobs.get_job(render["id"])["status"] == jobs.DONE


def test_crash_before_rename_recovery_original_intact(env, monkeypatch):
    event_id, render, stage, _ = render_ready(env, monkeypatch)
    original = Path.replace
    def crash(self, target):
        if "upload" in self.name:
            raise OSError("NAS disconnected")
        return original(self, target)
    monkeypatch.setattr(Path, "replace", crash)
    with pytest.raises(OSError):
        it.publish(render, stage)
    assert it.get_event(event_id)["state"] == "publishing"
    assert env.source.read_bytes() == b"original-video"
    jobs.update_job(render["id"], status=jobs.FAILED)
    it.recover(event_id)
    assert it.get_event(event_id)["state"] == "review"
    assert jobs.queue_render(it.get_event(event_id)["job_id"])


def test_crash_after_rename_recovery_detects_published_hash(env, monkeypatch):
    event_id, render, stage, _ = render_ready(env, monkeypatch)
    original = it.update_event
    def crash(event_id, **fields):
        if fields.get("state") == "published":
            raise OSError("process crashed")
        return original(event_id, **fields)
    monkeypatch.setattr(it, "update_event", crash)
    with pytest.raises(OSError):
        it.publish(render, stage)
    assert it.get_event(event_id)["state"] == "publishing"
    assert env.source.read_bytes() == b"cleaned-video"
    monkeypatch.setattr(it, "update_event", original)
    jobs.init_db()  # interrupted render becomes failed, journal survives
    it.recover(event_id)
    assert it.get_event(event_id)["state"] == "complete"


def test_recovery_never_overwrites_newer_import(env, monkeypatch):
    event_id, render, stage, _ = render_ready(env, monkeypatch)
    monkeypatch.setattr(it, "_rescan", lambda e: (_ for _ in ()).throw(ValueError("offline")))
    with pytest.raises(ValueError):
        it.publish(render, stage)
    env.source.write_bytes(b"newer-library-import")
    with pytest.raises(ValueError, match="differs"):
        it.recover(event_id)
    assert env.source.read_bytes() == b"newer-library-import"
    assert Path(it.get_event(event_id)["backup"]).read_bytes() == b"original-video"


def test_out_of_space_no_copy(env, monkeypatch):
    _, render, stage, _ = render_ready(env, monkeypatch)
    monkeypatch.setattr(it.shutil, "disk_usage", lambda p: SimpleNamespace(free=0))
    with pytest.raises(ValueError, match="space"):
        it.publish(render, stage)
    assert env.source.read_bytes() == b"original-video"


def test_retry_failed_scan_and_do_not_duplicate_render(env, monkeypatch):
    event_id = it.receive("radarr", payload())["event_id"]
    scan_id = it.get_event(event_id)["job_id"]
    jobs.update_job(scan_id, status=jobs.FAILED)
    retried = it.retry_scan(event_id)
    assert retried != scan_id and jobs.get_job(retried)["status"] == jobs.QUEUED
    with pytest.raises(ValueError):
        it.retry_scan(event_id)


def test_admin_connection_and_mapping_tests_are_read_only(env):
    assert env.client.post("/api/integrations/radarr/test", json={}).json["ok"]
    result = env.client.post("/api/integrations/path-test", json={"service": "radarr", "path": "/nas/movie.mkv"})
    assert result.json["path"] == str(env.source)
    assert not jobs.list_jobs() and not any(body for _, body in env.calls)


def test_snapshot_keeps_import_settings_and_full_validation(env, monkeypatch):
    _, render, _, _ = render_ready(env, monkeypatch)
    opts = json.loads(render["options"])
    assert opts["render_validation"] == "full"
    assert "--verify-render" in jobs.build_render_command(render)
    assert jobs.queue_render(render["parent_id"]) is None
    scan_edl = jobs.edl_path_for(render["parent_id"])
    before = Path(render["edl_path"]).read_bytes()
    EditDecisionList().to_json(scan_edl)
    assert Path(render["edl_path"]).read_bytes() == before
    assert Path(render["edl_path"]) != scan_edl


def test_upgrade_event_is_queued_behind_old_version(env):
    first = it.receive("radarr", payload())
    env.source.write_bytes(b"new-download")
    second = it.receive("radarr", payload(isUpgrade=True))
    assert first["job_id"] != second["job_id"]
    with pytest.raises(ValueError, match="changed"):
        it.preflight(jobs.get_job(first["job_id"]))
    it.preflight(jobs.get_job(second["job_id"]))


def test_partial_upload_failure_recoverable_without_touching_original(env, monkeypatch):
    event_id, render, stage, _ = render_ready(env, monkeypatch)
    def fail(src, target, expected):
        target.write_bytes(b"incomplete")
        raise OSError("disconnected")
    monkeypatch.setattr(it, "_copy_verified", fail)
    with pytest.raises(OSError):
        it.publish(render, stage)
    assert env.source.read_bytes() == b"original-video"
    it.recover(event_id)
    assert it.get_event(event_id)["state"] == "review"
    assert not Path(it.get_event(event_id)["partial"]).exists()


def test_cancel_or_delete_while_copying_prevents_replacement(env, monkeypatch):
    event_id, render, stage, _ = render_ready(env, monkeypatch)
    original = it._copy_verified
    def cancel(src, target, expected):
        original(src, target, expected)
        jobs.update_job(render["id"], status=jobs.CANCELED)
    monkeypatch.setattr(it, "_copy_verified", cancel)
    with pytest.raises(ValueError, match="Canceled"):
        it.publish(render, stage)
    assert env.source.read_bytes() == b"original-video"
    it.recover(event_id)


def test_owner_changed_or_untagged_during_render_prevents_publish(env, monkeypatch):
    _, render, stage, _ = render_ready(env, monkeypatch)
    monkeypatch.setattr(it, "arr", lambda service, resource, body=None:
                        {"tags": [5], "movieFile": {"id": 99}} if resource == "movie/1"
                        else env.api(service, resource, body))
    with pytest.raises(ValueError, match="upgraded"):
        it.publish(render, stage)
    assert env.source.read_bytes() == b"original-video"


def test_config_changes_cannot_redirect_existing_import(env, monkeypatch):
    _, render, stage, _ = render_ready(env, monkeypatch)
    it.save_config({"services": {"radarr": {"url": "http://127.0.0.1:9999"}}})
    with pytest.raises(ValueError, match="URL changed"):
        it.publish(render, stage)
    assert env.source.read_bytes() == b"original-video"


def test_failed_rescan_command_can_be_retried(env, monkeypatch):
    event_id = it.receive("radarr", payload())["event_id"]
    it.update_event(event_id, command_id=12)
    monkeypatch.setattr(it, "arr", lambda service, resource, body=None:
                        {"status": "failed"} if resource == "command/12" else env.api(service, resource, body))
    with pytest.raises(ValueError, match="rescan"):
        it._rescan(it.get_event(event_id))
    assert it.get_event(event_id)["command_id"] is None
    monkeypatch.setattr(it, "arr", env.api)
    it._rescan(it.get_event(event_id))
    assert it.get_event(event_id)["command_id"] == 12


@pytest.mark.parametrize("extension", ["mkv", "mp4"])
def test_real_media_replacement(env, monkeypatch, extension):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("Requires local FFmpeg and ffprobe")
    from cleancut.editor import Range, apply_cuts

    source = env.media / f"real.{extension}"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=128x72:r=24:d=4",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=4", "-c:v", "libx264",
                    "-c:a", "aac", "-shortest", str(source)], check=True)
    def api(service, resource, body=None):
        if resource == "moviefile/2":
            return {"movieId": 1, "path": f"/nas/real.{extension}"}
        return env.api(service, resource, body)
    monkeypatch.setattr(it, "arr", api)
    first = it.receive("radarr", payload())
    scan = jobs.get_job(first["job_id"])
    it.preflight(scan)
    EditDecisionList(decisions=[EditDecision(1, 2, "cut", "nudity")]).to_json(jobs.edl_path_for(scan["id"]))
    jobs._finish_ok(scan["id"], scan)
    render = jobs.get_job(jobs.queue_render(scan["id"]))
    jobs.update_job(render["id"], status=jobs.RUNNING)
    it.preflight(render)
    stage = jobs.render_stage_path(render)
    apply_cuts(source, [Range(1, 2)], stage, encoder="libx264", validation="full")
    jobs._finish_ok(render["id"], render)
    assert jobs.get_job(render["id"])["status"] == jobs.DONE
    assert abs(probe.probe_duration(source) - 3) < .15
    media_validation.validate_media(source, mode="full")
    assert it.get_event(first["event_id"])["state"] == "complete"
