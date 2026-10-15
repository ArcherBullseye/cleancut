"""Opt-in Arr imports and journaled, same-container library replacement.

No network connections or file publication happen merely by importing this module.
Credentials live separately from job options; the durable journal outlives jobs.
"""
from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import secrets
import shutil
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

from webapp import jobs, library, settings
from webapp.paths import DATA_DIR

CONFIG_PATH = DATA_DIR / "integrations.json"
_lock = threading.RLock()
SERVICES = ("radarr", "sonarr")
SUPPORTED_CONTAINERS = {".mp4", ".mkv", ".m4v", ".mov"}


def config() -> dict:
    result = {"enabled": False, "auto_render": False, "token": "", "services": {
        name: {"url": "", "api_key": "", "server_root": "/nas",
               "local_root": "/Volumes/Public", "tag": "cleancut"} for name in SERVICES}}
    if CONFIG_PATH.exists():
        stored = json.loads(CONFIG_PATH.read_text())
        result.update(stored)
    return result


def public_config() -> dict:
    cfg = config()
    cfg["token_configured"] = bool(cfg.pop("token"))
    for service in cfg["services"].values():
        service["key_configured"] = bool(service.pop("api_key"))
    return cfg


def local_url(raw: str) -> str:
    """Literal LAN/loopback addresses only: no DNS rebinding or cloud endpoints."""
    parsed = urlsplit(raw)
    try:
        address = ipaddress.ip_address(parsed.hostname or "")
        allowed = address.is_loopback or any(address in network for network in (
            ipaddress.ip_network("10.0.0.0/8"), ipaddress.ip_network("172.16.0.0/12"),
            ipaddress.ip_network("192.168.0.0/16"), ipaddress.ip_network("fc00::/7")))
        port = parsed.port
    except ValueError:
        raise ValueError("Use a literal private LAN or loopback IP address.") from None
    if (not allowed or parsed.scheme not in {"http", "https"} or parsed.username
            or parsed.password or parsed.query or parsed.fragment or port == 0):
        raise ValueError("Use an HTTP(S) LAN URL without credentials, query or fragment.")
    return raw.rstrip("/")


def save_config(body: dict) -> dict:
    if not isinstance(body, dict):
        raise ValueError("Expected integration settings.")  # noqa: TRY004 - API validation error
    with _lock:
        cfg = config()
        for key in ("enabled", "auto_render"):
            if key in body:
                if type(body[key]) is not bool:
                    raise ValueError(f"{key} must be true or false.")
                cfg[key] = body[key]
        services = body.get("services", {})
        if not isinstance(services, dict) or set(services) - set(SERVICES):
            raise ValueError("Unknown service.")
        for name, update in services.items():
            if not isinstance(update, dict):
                raise ValueError("Invalid service settings.")  # noqa: TRY004
            for key in ("url", "api_key", "server_root", "local_root", "tag"):
                if key not in update:
                    continue
                value = update[key]
                if not isinstance(value, str) or len(value) > 1024:
                    raise ValueError(f"Invalid {key}.")
                value = value.strip()
                if key == "api_key" and not value:  # blank preserves saved secret
                    continue
                if key == "url" and value:
                    value = local_url(value)
                if key in {"server_root", "local_root"} and (
                    not value.startswith("/") or ".." in Path(value).parts or value == "/"
                ):
                    raise ValueError("Choose an absolute media subdirectory, not /.")
                if key == "tag" and not value:
                    raise ValueError("An opt-in tag is required.")
                cfg["services"][name][key] = value
        if not cfg["token"]:
            cfg["token"] = secrets.token_urlsafe(32)
        if body.get("rotate_token") is True:
            cfg["token"] = secrets.token_urlsafe(32)
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = CONFIG_PATH.with_suffix(".tmp")
        # Set permissions before writing credentials, including the first save.
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        os.chmod(tmp, 0o600)
        with os.fdopen(fd, "w") as handle:
            json.dump(cfg, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        tmp.replace(CONFIG_PATH)
    return public_config()


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def arr(service: dict, resource: str, body: dict | None = None):
    url = local_url(service["url"]) + "/api/v3/" + resource
    if not service["api_key"]:
        raise ValueError("Enter the service API key in Integrations first.")
    request = urllib.request.Request(url, headers={"X-Api-Key": service["api_key"],
                                                 "Content-Type": "application/json"},
                                     data=json.dumps(body).encode() if body is not None else None)
    # No environment proxies, redirects, or credentials in exception messages.
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
        with opener.open(request, timeout=15) as response:
            return json.load(response)
    except (OSError, ValueError, urllib.error.URLError):
        raise ValueError("Arr API request failed. Check address, API key and connectivity.") from None


def map_path(service: dict, raw: str) -> Path:
    if not isinstance(raw, str) or "\\" in raw or ".." in PurePosixPath(raw).parts:
        raise ValueError("Invalid server media path.")
    try:
        relative = PurePosixPath(raw).relative_to(PurePosixPath(service["server_root"]))
    except ValueError:
        raise ValueError("Imported path is outside the configured server root.") from None
    root = Path(service["local_root"])
    if not root.is_dir():
        raise ValueError("NAS mount is unavailable; no directories will be created.")
    candidate = root.joinpath(*relative.parts)
    resolved = library.resolve_safe(str(candidate))
    if (resolved is None or not resolved.is_relative_to(root.resolve()) or candidate.is_symlink()
            or resolved != candidate.absolute() or not resolved.is_file()
            or resolved.suffix.lower() not in SUPPORTED_CONTAINERS):
        raise ValueError("Source must be an existing MP4/MKV/M4V/MOV inside the mapped media root, without symlinks.")
    if not os.access(resolved, os.R_OK) or not os.access(resolved.parent, os.W_OK):
        raise ValueError("NAS source must be readable and its folder writable.")
    return resolved


def fingerprint(path: Path) -> str:
    st = path.stat()
    return json.dumps([st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns])


def digest(path: Path) -> str:
    before = fingerprint(path)
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            sha.update(chunk)
    if fingerprint(path) != before:
        raise ValueError("File changed while reading; refusing replacement.")
    return sha.hexdigest()


def init_db() -> None:
    with jobs._connect() as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS imports (
            id INTEGER PRIMARY KEY, event_key TEXT UNIQUE NOT NULL,
            service TEXT NOT NULL, service_url TEXT NOT NULL, owner_id INTEGER NOT NULL, file_id INTEGER NOT NULL,
            remote_path TEXT NOT NULL, local_path TEXT NOT NULL, fingerprint TEXT NOT NULL,
            source_hash TEXT NOT NULL DEFAULT '', output_hash TEXT NOT NULL DEFAULT '',
            job_id INTEGER, render_id INTEGER, state TEXT NOT NULL DEFAULT 'queued',
            backup TEXT NOT NULL DEFAULT '', partial TEXT NOT NULL DEFAULT '',
            command_id INTEGER, error TEXT NOT NULL DEFAULT '', created_at REAL NOT NULL
        )""")


def get_event(event_id: int) -> dict:
    with jobs._connect() as conn:
        row = conn.execute("SELECT * FROM imports WHERE id=?", (event_id,)).fetchone()
    if row is None:
        raise ValueError("No such import.")
    return dict(row)


def update_event(event_id: int, **fields) -> None:
    allowed = {"source_hash", "output_hash", "job_id", "render_id", "state", "backup",
               "partial", "command_id", "error"}
    if not fields or set(fields) - allowed:
        raise ValueError("Invalid journal update.")
    with jobs._connect() as conn:
        conn.execute("UPDATE imports SET " + ",".join(f"{key}=?" for key in fields) + " WHERE id=?",
                     [*fields.values(), event_id])


def verify_owner(name: str, service: dict, owner_id: int, file_id: int) -> str:
    owner_kind, file_kind = ("movie", "moviefile") if name == "radarr" else ("series", "episodefile")
    owner = arr(service, f"{owner_kind}/{owner_id}")
    tags = arr(service, "tag")
    permitted = {tag["id"] for tag in tags if tag["label"] == service["tag"]}
    if not permitted.intersection(owner.get("tags", [])):
        raise ValueError("Movie/series does not have the configured CleanCut opt-in tag.")
    media = arr(service, f"{file_kind}/{file_id}")
    if media.get(owner_kind + "Id") != owner_id:
        raise ValueError("File no longer belongs to this movie/series.")
    if name == "radarr" and owner.get("movieFile", {}).get("id") != file_id:
        raise ValueError("Movie has been upgraded or moved.")
    return media["path"]


def receive(name: str, payload: dict) -> dict:
    cfg = config()
    if not cfg["enabled"]:
        raise ValueError("Integration is disabled.")
    if not isinstance(payload, dict):
        raise ValueError("Expected an Arr webhook object.")  # noqa: TRY004
    if payload.get("eventType") == "Test":
        return {"ok": True, "test": True}
    if payload.get("eventType") != "Download":
        return {"ok": True, "ignored": True}
    owner_kind, file_kind = ("movie", "movieFile") if name == "radarr" else ("series", "episodeFile")
    owner_id = payload.get(owner_kind, {}).get("id") if isinstance(payload.get(owner_kind), dict) else None
    file_id = payload.get(file_kind, {}).get("id") if isinstance(payload.get(file_kind), dict) else None
    if type(owner_id) is not int or type(file_id) is not int or min(owner_id, file_id) <= 0:
        raise ValueError("Import must identify one movie/episode file and its owner.")
    service = cfg["services"][name]
    remote = verify_owner(name, service, owner_id, file_id)
    video = map_path(service, remote)  # Never trust the webhook's filesystem path.
    fp = fingerprint(video)
    key = hashlib.sha256(json.dumps([name, service["url"], file_id, remote, fp]).encode()).hexdigest()
    with _lock, jobs._connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        existing = conn.execute("SELECT * FROM imports WHERE event_key=?", (key,)).fetchone()
        if existing:
            return {"ok": True, "event_id": existing["id"], "job_id": existing["job_id"], "existing": True}
        # A rescan callback must not clean an already published copy a second time.
        previous = conn.execute("SELECT * FROM imports WHERE local_path=? AND state IN "
                                "('publishing','published','complete') ORDER BY id DESC LIMIT 1", (str(video),)).fetchone()
        if previous and previous["output_hash"] and digest(video) == previous["output_hash"]:
            return {"ok": True, "event_id": previous["id"], "job_id": previous["job_id"], "existing": True}
        # An upgrade at the same path is a NEW source fingerprint. Queue it
        # behind the old job; the old job's source checks will refuse to publish.
        row = conn.execute("INSERT INTO imports (event_key,service,service_url,owner_id,file_id,remote_path,local_path,"
                           "fingerprint,created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                           (key, name, service["url"], owner_id, file_id, remote, str(video), fp, time.time()))
        event_id = row.lastrowid
        scan_cfg = settings.load()
        opts = {k: v for k, v in scan_cfg.items() if k != "speech_token"}
        opts.update(integration_id=event_id, auto_render=cfg["auto_render"], output_location="source",
                    render_validation="full", use_llm=scan_cfg["local_ai_enabled"],
                    use_vlm=scan_cfg["local_ai_enabled"])
        row = conn.execute("INSERT INTO jobs (kind,status,video_path,title,preset,options,created_at) "
                           "VALUES ('scan',?,?,?,?,?,?)", (jobs.QUEUED, str(video), video.stem,
                            scan_cfg["preset"], json.dumps(opts), time.time()))
        job_id = row.lastrowid
        jobs.job_dir(job_id).mkdir(parents=True, exist_ok=True)
        conn.execute("UPDATE imports SET job_id=? WHERE id=?", (job_id, event_id))
    jobs._wake.set()
    return {"ok": True, "event_id": event_id, "job_id": job_id, "existing": False}


def check_source(event: dict) -> Path:
    cfg = config()
    if not cfg["enabled"]:
        raise ValueError("Integration disabled; source left untouched.")
    service = cfg["services"][event["service"]]
    if service["url"] != event["service_url"]:
        raise ValueError("Service URL changed since import; restore its settings before continuing.")
    remote = verify_owner(event["service"], service, event["owner_id"], event["file_id"])
    source = map_path(service, remote)
    if (remote != event["remote_path"] or str(source) != event["local_path"]
            or fingerprint(source) != event["fingerprint"]):
        raise ValueError("Source changed, moved or was upgraded since import; refusing replacement.")
    return source


def preflight(job: dict) -> None:
    event_id = json.loads(job["options"]).get("integration_id")
    if not event_id:
        return
    event = get_event(event_id)
    if event["state"] not in {"queued", "scanning", "review", "rendering"}:
        raise ValueError("Import already published or requires recovery; refusing to process it again.")
    source = check_source(event)
    hashed = digest(source)
    if event["source_hash"] and hashed != event["source_hash"]:
        raise ValueError("Source content changed; refusing replacement.")
    update_event(event_id, source_hash=hashed, state="scanning" if job["kind"] == "scan" else "rendering")


def _copy_verified(source: Path, target: Path, expected_hash: str) -> None:
    # Exclusive creation protects unrelated/recovery files, even after a crash.
    with source.open("rb") as src, target.open("xb") as dst:
        shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
        dst.flush()
        os.fsync(dst.fileno())
    if digest(target) != expected_hash:
        raise ValueError("NAS copy verification failed; original and recovery files retained.")


def _rescan(event: dict) -> None:
    service = config()["services"][event["service"]]
    if service["url"] != event["service_url"]:
        raise ValueError("Service URL changed; restore it before recovering this import.")
    if not event["command_id"]:
        movie = event["service"] == "radarr"
        command = arr(service, "command", {"name": "RescanMovie" if movie else "RescanSeries",
                                           "movieId" if movie else "seriesId": event["owner_id"]})
        update_event(event["id"], command_id=command["id"])
        event["command_id"] = command["id"]
    # A pending/failed rescan retains the backup, and can be retried from the UI.
    for _ in range(30):
        command = arr(service, f"command/{event['command_id']}")
        if command.get("status") == "completed":
            return
        if command.get("status") in {"failed", "aborted", "cancelled"}:
            update_event(event["id"], command_id=None)
            break
        time.sleep(1)
    raise ValueError("Library rescan not confirmed. Recovery backup retained; retry recovery later.")


def finish_publication(event: dict) -> None:
    source = Path(event["local_path"])
    mapped = map_path(config()["services"][event["service"]], event["remote_path"])
    if mapped != source:
        raise ValueError("Mapping changed; restore it before recovery.")
    if digest(source) != event["output_hash"]:
        raise ValueError("Library file changed after publication. Backup retained for manual recovery.")
    _rescan(event)
    if digest(source) != event["output_hash"]:
        raise ValueError("Library file changed during rescan. Backup retained for manual recovery.")
    backup = Path(event["backup"])
    if backup.exists():
        if digest(backup) != event["source_hash"]:
            raise ValueError("Backup differs from original; retained for manual recovery.")
        backup.unlink()
    update_event(event["id"], state="complete", error="")


def publish(job: dict, stage: Path) -> None:
    from cleancut.editor_ranges import Range, keep_segments, normalize_cuts
    from cleancut.edl import EditDecisionList
    from cleancut.media_validation import validate_media
    from cleancut.probe import probe_duration, probe_streams, video_stream

    event_id = json.loads(job["options"])["integration_id"]
    with _lock:
        event = get_event(event_id)
        if event["state"] != "rendering":
            raise ValueError("Import is not ready for publication.")
        source = check_source(event)
        if digest(source) != event["source_hash"]:
            raise ValueError("Original changed during render; refusing replacement.")
        if stage.suffix.lower() != source.suffix.lower():
            raise ValueError("Replacement container must match the original extension.")
        duration = probe_duration(source)
        edl = EditDecisionList.from_json(Path(job["edl_path"]))
        cuts = normalize_cuts([Range(d.start, d.end) for d in edl.by_action("cut")], duration)
        expected = sum(segment.duration for segment in keep_segments(duration, cuts))
        stream = video_stream(probe_streams(source))
        if not stream or expected <= 0:
            raise ValueError("Invalid source or empty output timeline.")
        validate_media(stage, mode="full", expected_duration=expected,
                       expected_dimensions=(stream.width, stream.height))
        if (any(s.codec_type == "audio" for s in probe_streams(source))
                and not any(s.codec_type == "audio" for s in probe_streams(stage))):
            raise ValueError("Replacement is missing the source audio.")
        output_hash = digest(stage)
        backup = source.with_name(f".{source.name}.cleancut-{event_id}.backup")
        partial = source.with_name(f".{source.stem}.cleancut-{event_id}.upload{source.suffix}")
        required = source.stat().st_size + stage.stat().st_size + 64 * 1024 * 1024
        if shutil.disk_usage(source.parent).free < required:
            raise ValueError("Not enough NAS space for both verified upload and temporary original backup.")
        update_event(event_id, backup=str(backup), partial=str(partial), output_hash=output_hash, state="preparing")
        _copy_verified(stage, partial, output_hash)
        _copy_verified(source, backup, event["source_hash"])
        # Last API/identity/content check, after potentially long NAS copies.
        check_source(event)
        if digest(source) != event["source_hash"]:
            raise ValueError("Source changed during upload; refusing replacement.")
        if (jobs.get_job(job["id"]) or {}).get("status") != jobs.RUNNING:
            raise ValueError("Canceled before replacement. Original retained.")
        # Journal BEFORE the only destructive operation. os.replace is one
        # same-directory rename: no missing-file window, backup already durable.
        update_event(event_id, state="publishing")
        if fingerprint(source) != event["fingerprint"]:
            raise ValueError("Source changed immediately before replacement; original retained.")
        partial.replace(source)
        update_event(event_id, state="published")
        finish_publication(get_event(event_id))


def recover(event_id: int) -> None:
    """Explicit, conservative recovery; never overwrite a possibly newer import."""
    with _lock:
        event = get_event(event_id)
        if event["state"] not in {"preparing", "publishing", "published"}:
            raise ValueError("This import has no interrupted publication to recover.")
        source = Path(event["local_path"])
        mapped = map_path(config()["services"][event["service"]], event["remote_path"])
        if mapped != source:
            raise ValueError("Mapping changed; restore it before recovery.")
        actual = digest(source)
        if actual == event["output_hash"]:
            update_event(event_id, state="published")
            finish_publication(get_event(event_id))
            if event["render_id"] and jobs.get_job(event["render_id"]):
                jobs.update_job(event["render_id"], status=jobs.DONE, stage="Complete", error="", progress=100)
        elif actual == event["source_hash"]:
            # Original is verified intact, so incomplete copies from this
            # publication attempt are dispensable too. Never touch other files.
            for key in ("partial", "backup"):
                path = Path(event[key])
                if path.exists():
                    path.unlink()
            update_event(event_id, state="review", output_hash="", error="Original intact; render can be queued again.")
        else:
            raise ValueError("Source differs from both versions. All recovery files retained; manual inspection required.")


def retry_scan(event_id: int) -> int:
    with _lock:
        event = get_event(event_id)
        if event["state"] not in {"queued", "scanning"}:
            raise ValueError("Review/render the existing scan or recover its publication instead.")
        old = jobs.get_job(event["job_id"])
        if not old or old["status"] not in {jobs.FAILED, jobs.CANCELED}:
            raise ValueError("Only a failed/canceled scan with retained job details can be retried.")
        source = check_source(event)
        if jobs.active_job_for(str(source)):
            raise ValueError("This source already has an active job.")
        job_id = jobs.create_job("scan", str(source), title=old["title"], preset=old["preset"],
                                 options=json.loads(old["options"]), wake=False)
        update_event(event_id, job_id=job_id, state="queued", error="")
        jobs._wake.set()
        return job_id


def queue_render(scan: dict, options: dict) -> int | None:
    """Commit the render, frozen EDL and journal link before the worker sees it."""
    with _lock, jobs._connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        event_id = options["integration_id"]
        event = conn.execute("SELECT * FROM imports WHERE id=?", (event_id,)).fetchone()
        if not event or event["state"] not in {"review", "rendering"}:
            return None
        if conn.execute("SELECT id FROM jobs WHERE video_path=? AND status IN (?,?)",
                        (scan["video_path"], jobs.QUEUED, jobs.RUNNING)).fetchone():
            return None
        row = conn.execute("INSERT INTO jobs (kind,status,video_path,title,preset,options,output_path,parent_id,created_at) "
                           "VALUES ('render',?,?,?,?,?,?,?,?)", (jobs.QUEUED, scan["video_path"], scan["title"],
                            scan["preset"], json.dumps(options), scan["video_path"], scan["id"], time.time()))
        job_id = row.lastrowid
        directory = jobs.job_dir(job_id)
        directory.mkdir(parents=True, exist_ok=True)
        frozen_edl = directory / "edl.json"
        from cleancut.edl import EditDecisionList
        EditDecisionList.from_json(Path(scan["edl_path"])).to_json(frozen_edl)
        conn.execute("UPDATE jobs SET edl_path=? WHERE id=?", (str(frozen_edl), job_id))
        conn.execute("UPDATE imports SET render_id=? WHERE id=?", (job_id, event_id))
    jobs._wake.set()
    return job_id


def events() -> list[dict]:
    with jobs._connect() as conn:
        rows = [dict(row) for row in conn.execute("SELECT * FROM imports ORDER BY id DESC LIMIT 200")]
    for row in rows:
        job = jobs.get_job(row["render_id"] or row["job_id"]) if row["job_id"] else None
        row["job_status"] = job["status"] if job else "deleted"
        row["job_error"] = job["error"] if job else ""
    return rows


def protected(job_id: int) -> bool:
    with jobs._connect() as conn:
        if not conn.execute("SELECT name FROM sqlite_master WHERE name='imports'").fetchone():
            return False
        return conn.execute("SELECT id FROM imports WHERE (job_id=? OR render_id=?) "
                            "AND state IN ('preparing','publishing','published')", (job_id, job_id)).fetchone() is not None
