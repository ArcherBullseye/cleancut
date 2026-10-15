"""cleancut for Umbrel -- web front-end over the cleancut pipeline."""

from __future__ import annotations

import ipaddress
import json
import os
import platform
import secrets
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from flask import Flask, jsonify, render_template, request, send_file

from webapp import integrations, jobs, library, review
from webapp import settings as settings_store
from webapp.paths import OUTPUT_DIR, ensure_dirs, media_roots

APP_VERSION = os.environ.get("CLEANCUT_VERSION", "2.0.0-mac-beta.15")

app = Flask(__name__, template_folder="../templates", static_folder="../static")
app.config["JSON_SORT_KEYS"] = False


def _bad(message: str, code: int = 400):
    return jsonify({"ok": False, "error": message}), code


def _job_or_404(job_id: int):
    job = jobs.get_job(job_id)
    if job is None:
        return None, _bad("No such job.", 404)
    return job, None


def _video_for(job: dict[str, Any]) -> Path | None:
    event_id = json.loads(job["options"] or "{}").get("integration_id")
    if event_id and integrations.get_event(event_id)["state"] in {"publishing", "published", "complete"}:
        # Review timestamps describe the original, not the replaced timeline.
        return None
    return library.resolve_safe(job["video_path"])


# --------------------------------------------------------------------------
# pages
# --------------------------------------------------------------------------

@app.route("/")
def page_library():
    return render_template("library.html", version=APP_VERSION, page="library")


@app.route("/jobs")
def page_jobs():
    return render_template("jobs.html", version=APP_VERSION, page="jobs")


@app.route("/job/<int:job_id>")
def page_job(job_id: int):
    job = jobs.get_job(job_id)
    if job is None:
        return render_template("missing.html", version=APP_VERSION, page="jobs"), 404
    return render_template("job.html", version=APP_VERSION, page="jobs", job=job)


@app.route("/settings")
def page_settings():
    return render_template("settings.html", version=APP_VERSION, page="settings")


@app.route("/integrations")
def page_integrations():
    return render_template("integrations.html", version=APP_VERSION, page="integrations")


def _integration_admin():
    # Admin credentials and replacement controls stay on the Mac itself.
    # LAN webhooks have separate authentication; never trust forwarded headers.
    try:
        if not ipaddress.ip_address(request.remote_addr).is_loopback:
            return _bad("Open Integrations on the CleanCut Mac using localhost (or an SSH tunnel).", 403)
    except ValueError:
        return _bad("Local access required.", 403)
    if urlsplit(request.host_url).hostname not in {"localhost", "127.0.0.1", "::1"}:
        return _bad("Use localhost for integration administration.", 403)
    if request.method != "GET":
        origin = request.headers.get("Origin")
        if origin and urlsplit(origin).netloc != request.host:
            return _bad("Cross-origin request refused.", 403)
        if not request.is_json:
            return _bad("Use application/json.", 415)
    return None


@app.route("/api/integrations", methods=["GET", "POST"])
def api_integrations():
    denied = _integration_admin()
    if denied:
        return denied
    try:
        cfg = integrations.save_config(request.get_json(silent=True)) if request.method == "POST" else integrations.public_config()
        return jsonify({"settings": cfg, "events": integrations.events()})
    except (ValueError, OSError) as exc:
        return _bad(str(exc))


@app.route("/api/integrations/token", methods=["POST"])
def api_integration_token():
    denied = _integration_admin()
    if denied:
        return denied
    response = jsonify({"token": integrations.config()["token"]})
    response.headers["Cache-Control"] = "no-store"
    return response


@app.route("/api/integrations/<name>/test", methods=["POST"])
def api_integration_test(name):
    denied = _integration_admin()
    if denied:
        return denied
    if name not in integrations.SERVICES:
        return _bad("Unknown service.")
    try:
        service = integrations.config()["services"][name]
        status = integrations.arr(service, "system/status")
        roots = integrations.arr(service, "rootfolder")
        return jsonify({"ok": True, "version": status.get("version"),
                        "roots": [r["path"] for r in roots],
                        "mount_available": Path(service["local_root"]).is_dir()})
    except (ValueError, OSError) as exc:
        return _bad(str(exc))


@app.route("/api/integrations/path-test", methods=["POST"])
def api_integration_path_test():
    denied = _integration_admin()
    if denied:
        return denied
    body = request.get_json(silent=True)
    if not isinstance(body, dict) or body.get("service") not in integrations.SERVICES:
        return _bad("Choose a service and imported file path.")
    try:
        path = integrations.map_path(integrations.config()["services"][body["service"]], body.get("path"))
        return jsonify({"ok": True, "path": str(path)})
    except (ValueError, OSError) as exc:
        return _bad(str(exc))


@app.route("/api/integrations/<int:event_id>/recover", methods=["POST"])
def api_integration_recover(event_id):
    denied = _integration_admin()
    if denied:
        return denied
    try:
        integrations.recover(event_id)
        return jsonify({"ok": True})
    except (ValueError, OSError) as exc:
        return _bad(str(exc))


@app.route("/api/integrations/<int:event_id>/retry", methods=["POST"])
def api_integration_retry(event_id):
    denied = _integration_admin()
    if denied:
        return denied
    try:
        return jsonify({"ok": True, "job_id": integrations.retry_scan(event_id)})
    except (ValueError, OSError) as exc:
        return _bad(str(exc))


@app.route("/api/integrations/<name>/webhook", methods=["POST"])
def api_integration_webhook(name):
    if name not in integrations.SERVICES:
        return _bad("Unknown service.", 404)
    auth = request.authorization
    supplied = auth.password if auth and auth.type == "basic" and auth.username == "cleancut" else ""
    token = integrations.config()["token"]
    if not token or not secrets.compare_digest(supplied or "", token):
        return _bad("Invalid webhook credentials.", 401)
    if request.content_length is None or request.content_length > 1024 * 1024:
        return _bad("Webhook body too large or missing Content-Length.", 413)
    try:
        return jsonify(integrations.receive(name, request.get_json(silent=True)))
    except (ValueError, OSError) as exc:
        return _bad(str(exc), 409)


# --------------------------------------------------------------------------
# library
# --------------------------------------------------------------------------

@app.route("/api/browse")
def api_browse():
    raw = request.args.get("path", "")
    if not raw:
        roots = library.list_roots()
        if not roots:
            return jsonify({
                "roots": [], "path": "", "parent": None, "dirs": [], "files": [],
                "error": "No media roots are mounted. Check the app's docker-compose volumes.",
            })
        # A single root is the common case -- open straight into it.
        if len(roots) == 1:
            listing = library.list_dir(Path(roots[0]["path"]))
            listing["roots"] = roots
            return jsonify(listing)
        return jsonify({"roots": roots, "path": "", "parent": None,
                        "dirs": roots, "files": [], "error": None})

    path = library.resolve_safe(raw)
    if path is None or not path.is_dir():
        return _bad("That folder is outside the mounted media roots.", 403)
    listing = library.list_dir(path)
    listing["roots"] = library.list_roots()
    return jsonify(listing)


@app.route("/api/search")
def api_search():
    return jsonify({"results": library.search(request.args.get("q", ""))})


@app.route("/api/folder-videos", methods=["POST"])
def api_folder_videos():
    body = request.get_json(silent=True)
    if not isinstance(body, dict) or not isinstance(body.get("path"), str):
        return _bad("Choose a folder.")
    if type(body.get("recursive", True)) is not bool:
        return _bad("Include subfolders must be true or false.")
    path = library.resolve_safe(body["path"])
    if path is None or not path.is_dir():
        return _bad("That folder is outside the mounted media roots.", 403)
    try:
        files = library.folder_videos(path, body.get("recursive", True))
    except (OSError, ValueError) as exc:
        return _bad(str(exc))
    return jsonify({"files": files})


# --------------------------------------------------------------------------
# jobs
# --------------------------------------------------------------------------

@app.route("/api/jobs")
def api_jobs():
    state = request.args.get("state", "")
    if state and state not in {"queued", "running", "done", "failed", "canceled"}:
        return _bad("Unknown job state.")
    records = jobs.list_jobs(limit=10000)
    return jsonify({"jobs": [job for job in records if not state or job["status"] == state]})


@app.route("/api/jobs/delete", methods=["POST"])
def api_delete_jobs():
    body = request.get_json(silent=True) or {}
    ids = body.get("ids") if isinstance(body, dict) else None
    if not isinstance(ids, list) or not ids or len(ids) > 10000 or any(type(i) is not int for i in ids):
        return _bad("Choose jobs to delete.")
    removed, skipped = [], []
    for job_id in dict.fromkeys(ids):
        job = jobs.get_job(job_id)
        if not job:
            continue
        children = jobs.descendants(job_id)
        if job["status"] in jobs.ACTIVE_STATUSES or any(c["status"] in jobs.ACTIVE_STATUSES for c in children):
            skipped.append(job_id)
            continue
        try:
            jobs.delete_job(job_id)
        except ValueError:
            skipped.append(job_id)
            continue
        removed.append(job_id)
    return jsonify({"deleted": removed, "skipped": skipped})


@app.route("/api/scan", methods=["POST"])
def api_scan():
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return _bad("Choose one or more videos.")
    raw_paths = body.get("paths", [body.get("path", "")])
    if (not isinstance(raw_paths, list) or not 1 <= len(raw_paths) <= library.MAX_BATCH_FILES
            or any(not isinstance(path, str) or not path for path in raw_paths)):
        return _bad(f"Choose between 1 and {library.MAX_BATCH_FILES} videos.")
    videos = []
    for raw_path in raw_paths:
        video = library.resolve_safe(raw_path)
        if video is None or not video.is_file():
            return _bad("That file is outside the mounted media roots or no longer available.", 403)
        if not os.access(video, os.R_OK):
            return _bad("That video is not readable. Check the NAS share permissions.", 403)
        if not library.is_video(video):
            return _bad("Not a recognised video file.")
        if video not in videos:
            videos.append(video)

    existing = jobs.active_job_for(str(videos[0])) if len(videos) == 1 else None
    if existing and "paths" not in body:
        return jsonify({"ok": True, "job_id": existing["id"], "existing": True})

    cfg = settings_store.load()
    preset = body.get("preset") or cfg["preset"]
    if preset not in ("fast", "balanced", "thorough"):
        return _bad("Unknown preset.")

    categories = body.get("categories", cfg["categories"])
    actions = body.get("actions", cfg["actions"])
    if not isinstance(categories, list) or any(cat not in review.CATEGORIES for cat in categories):
        return _bad("Unknown categories.")
    if not isinstance(actions, dict) or any(
        cat not in review.CATEGORIES or action not in review.ACTIONS
        or (action == "replace" and cat != "profanity") for cat, action in actions.items()
    ):
        return _bad("Unknown actions. Replace is only supported for profanity words.")
    actions = {**cfg["actions"], **actions}
    word_padding = body.get("word_end_padding_ms", cfg["word_end_padding_ms"])
    if type(word_padding) is not int or not 0 <= word_padding <= 500:
        return _bad("Word ending buffer must be between 0 and 500 milliseconds.")

    local_ai = bool(body.get("use_local_ai", cfg["local_ai_enabled"]))
    options: dict[str, Any] = {
        "categories": categories,
        "actions": actions,
        "ollama_host": body.get("ollama_host", cfg["ollama_host"]),
        "llm_model": cfg["llm_model"],
        "vlm_model": cfg["vlm_model"],
        "prefer_language": body.get("prefer_language") or cfg["prefer_language"],
        "output_dir": cfg["output_dir"],
        "output_location": cfg["output_location"],
        "auto_render": bool(body.get("auto_render", cfg["auto_render"])),
        "use_visual": bool(body.get("use_visual", cfg["use_visual"])),
        "allow_solo_visual": bool(body.get("allow_solo_visual", cfg["allow_solo_visual"])),
        "word_end_padding_ms": word_padding,
        "analysis_height": cfg["analysis_height"],
        "analysis_proxy": cfg["analysis_proxy"],
        "nudity_model": cfg["nudity_model"],
        "use_llm": local_ai,
        "use_vlm": local_ai,
    }
    for key in ("use_llm", "use_vlm", "use_audio_events"):
        if key in body:
            options[key] = bool(body[key])

    queued = jobs.create_scan_jobs(videos, preset=preset, options=options, group=len(videos) > 1)
    # Remember this form for future jobs; the queued job owns its independent
    # snapshot. No global render-time switch can turn its Mute into Replace.
    settings_store.save({
        "preset": preset, "categories": categories, "actions": actions,
        "prefer_language": options["prefer_language"], "auto_render": options["auto_render"],
        "local_ai_enabled": local_ai, "use_visual": options["use_visual"],
        "allow_solo_visual": options["allow_solo_visual"],
        "word_end_padding_ms": word_padding,
    })
    return jsonify({"ok": True, "job_id": queued[0]["job_id"], "jobs": queued,
                    "batch_id": next((item["batch_id"] for item in queued if item.get("batch_id")), None),
                    "queued_count": sum(not item["existing"] for item in queued)})


@app.route("/api/job/<int:job_id>")
def api_job(job_id: int):
    job, err = _job_or_404(job_id)
    if err:
        return err
    payload = dict(job)
    payload["options"] = json.loads(job["options"] or "{}")
    if payload["options"].get("integration_id"):
        payload["integration_state"] = integrations.get_event(payload["options"]["integration_id"])["state"]
    if job["kind"] == "scan" and job["edl_path"] and Path(job["edl_path"]).exists():
        try:
            payload["summary"] = review.summarize(review.load_edl(job["edl_path"]))
        except (OSError, json.JSONDecodeError):
            payload["summary"] = None
    if job["output_path"]:
        out = Path(job["output_path"])
        payload["output_exists"] = out.exists()
        payload["output_size"] = out.stat().st_size if out.exists() else 0
    return jsonify(payload)


@app.route("/api/job/<int:job_id>/log")
def api_job_log(job_id: int):
    job, err = _job_or_404(job_id)
    if err:
        return err
    return jsonify({"log": jobs.read_log(job_id)})


@app.route("/api/job/<int:job_id>/cancel", methods=["POST"])
def api_job_cancel(job_id: int):
    return jsonify({"ok": jobs.cancel_job(job_id)})


@app.route("/api/job/<int:job_id>", methods=["DELETE"])
def api_job_delete(job_id: int):
    try:
        jobs.delete_job(job_id)
    except ValueError as exc:
        return _bad(str(exc), 409)
    return jsonify({"ok": True})


@app.route("/api/job/<int:job_id>/render", methods=["POST"])
def api_job_render(job_id: int):
    job, err = _job_or_404(job_id)
    if err:
        return err
    if job["kind"] != "scan" or not job["edl_path"]:
        return _bad("Only a finished scan can be rendered.")
    body = request.get_json(silent=True) or {}
    overrides: dict[str, Any] = {}
    if body.get("subtitle_mode") in ("soft", "burn", "none"):
        overrides["subtitle_mode"] = body["subtitle_mode"]
    if body.get("quality") is not None:
        overrides["quality"] = int(body["quality"])
    if body.get("encoder") in (
        "auto", "videotoolbox", "hevc_videotoolbox", "libx264", "libx265"
    ):
        overrides["encoder"] = body["encoder"]
    if body.get("render_validation") in ("none", "quick", "full"):
        overrides["render_validation"] = body["render_validation"]
    render_id = jobs.queue_render(job_id, overrides=overrides)
    if render_id is None:
        if json.loads(job["options"] or "{}").get("integration_id"):
            return _bad("Import already has active work, was published, or needs recovery. Check Integrations and the NAS mount.", 409)
        return _bad(
            "Could not queue the render. Make sure the source/NAS folder is mounted "
            "and writable, or choose a writable output folder in Settings."
        )
    return jsonify({"ok": True, "job_id": render_id})


@app.route("/api/job/<int:job_id>/download")
def api_job_download(job_id: int):
    job, err = _job_or_404(job_id)
    if err:
        return err
    if not job["output_path"]:
        return _bad("This job has no output file.", 404)
    out = Path(job["output_path"])
    if not out.exists():
        return _bad("The output file is gone.", 404)
    # conditional=True so the browser can seek in the video rather than
    # downloading a multi-gigabyte file just to check the edit.
    mime = {".mkv": "video/x-matroska", ".mov": "video/quicktime", ".m4v": "video/x-m4v"}.get(out.suffix.lower(), "video/mp4")
    return send_file(out, as_attachment=False, conditional=True,
                     download_name=out.name, mimetype=mime)


# --------------------------------------------------------------------------
# review
# --------------------------------------------------------------------------

def _edl_job(job_id: int):
    job, err = _job_or_404(job_id)
    if err:
        return None, None, err
    if not job["edl_path"] or not Path(job["edl_path"]).exists():
        return None, None, _bad("This job has no EDL yet.", 404)
    return job, Path(job["edl_path"]), None


@app.route("/api/job/<int:job_id>/edl")
def api_edl(job_id: int):
    job, path, err = _edl_job(job_id)
    if err:
        return err
    data = review.load_edl(path)
    return jsonify({"edl": data, "summary": review.summarize(data), "video": job["video_path"]})


@app.route("/api/job/<int:job_id>/edl/decision", methods=["POST"])
def api_edl_decision(job_id: int):
    job, path, err = _edl_job(job_id)
    if err:
        return err
    body = request.get_json(silent=True) or {}
    try:
        index = int(body.get("index", -1))
    except (TypeError, ValueError):
        return _bad("index must be a number.")
    data = review.load_edl(path)
    try:
        decision = review.apply_edit(data, index, body)
    except (IndexError, ValueError) as e:
        return _bad(str(e))
    review.save_edl(path, data)
    return jsonify({"ok": True, "decision": decision, "summary": review.summarize(data)})


@app.route("/api/job/<int:job_id>/edl/bulk", methods=["POST"])
def api_edl_bulk(job_id: int):
    job, path, err = _edl_job(job_id)
    if err:
        return err
    body = request.get_json(silent=True) or {}
    data = review.load_edl(path)
    changed = review.set_all_accepted(
        data, bool(body.get("accepted", True)), body.get("category") or None
    )
    review.save_edl(path, data)
    return jsonify({"ok": True, "changed": changed, "summary": review.summarize(data)})


@app.route("/api/job/<int:job_id>/edl/add", methods=["POST"])
def api_edl_add(job_id: int):
    job, path, err = _edl_job(job_id)
    if err:
        return err
    body = request.get_json(silent=True) or {}
    data = review.load_edl(path)
    try:
        review.add_decision(
            data,
            review.parse_timestamp(body.get("start", "")),
            review.parse_timestamp(body.get("end", "")),
            body.get("category", "sex"),
            body.get("action", "cut"),
            body.get("reason", ""),
        )
    except (ValueError, TypeError) as e:
        return _bad(str(e))
    review.save_edl(path, data)
    return jsonify({"ok": True, "summary": review.summarize(data)})


@app.route("/api/job/<int:job_id>/thumb")
def api_thumb(job_id: int):
    job, err = _job_or_404(job_id)
    if err:
        return err
    video = _video_for(job)
    if video is None or not video.exists():
        return _bad("Source video not found.", 404)
    try:
        at = float(request.args.get("t", "0"))
    except ValueError:
        return _bad("t must be a number.")
    out = review.thumbnail(job_id, video, at)
    if out is None:
        return _bad("Could not extract a frame.", 500)
    return send_file(out, mimetype="image/jpeg", conditional=True)


@app.route("/api/job/<int:job_id>/clip")
def api_clip(job_id: int):
    job, err = _job_or_404(job_id)
    if err:
        return err
    video = _video_for(job)
    if video is None or not video.exists():
        return _bad("Source video not found.", 404)
    try:
        start = float(request.args.get("start", "0"))
        end = float(request.args.get("end", "0"))
    except ValueError:
        return _bad("start and end must be numbers.")
    out = review.clip(job_id, video, start, end)
    if out is None:
        return _bad("Could not build a preview clip.", 500)
    return send_file(out, mimetype="video/mp4", conditional=True)


# --------------------------------------------------------------------------
# settings and health
# --------------------------------------------------------------------------

@app.route("/api/settings", methods=["GET", "POST"])
def api_settings():
    if request.method == "POST":
        body = request.get_json(silent=True) or {}
        return jsonify({"ok": True, "settings": settings_store.save(body)})
    return jsonify({"settings": settings_store.load()})


@app.route("/api/ollama")
def api_ollama():
    """Report whether Ollama is reachable and which models it has.

    The LLM and VLM signals silently produce nothing when Ollama is missing a
    model, so surfacing this up front is the difference between a scan that
    quietly skips half its detectors and one the user can trust.
    """
    import urllib.error
    import urllib.request

    host = (request.args.get("host") or settings_store.load()["ollama_host"]).strip()
    if not host:
        return jsonify({"ok": False, "reason": "No Ollama host configured.", "models": []})
    try:
        with urllib.request.urlopen(f"{host.rstrip('/')}/api/tags", timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        models = sorted(m.get("name", "") for m in data.get("models", []))
        return jsonify({"ok": True, "host": host, "models": models})
    except (urllib.error.URLError, OSError, json.JSONDecodeError, ValueError) as e:
        return jsonify({"ok": False, "host": host, "reason": str(e), "models": []})


def _speech_config(updates: dict | None = None):
    from cleancut.config import Config

    cfg = settings_store.load()
    for key in ("speech_host", "speech_model", "speech_token"):
        if updates and key in updates:
            cfg[key] = updates[key]
    return Config(profanity_audio="replace", speech_host=cfg["speech_host"],
                  speech_model=cfg["speech_model"], speech_token=cfg["speech_token"])


@app.route("/api/speech", methods=["POST"])
def api_speech():
    from cleancut.speech import check_service

    try:
        health = check_service(_speech_config(request.get_json(silent=True) or {}))
        return jsonify(ok=True, model=health["model"], model_loaded=health.get("model_loaded", False))
    except Exception as exc:
        return jsonify(ok=False, reason=str(exc))


@app.route("/api/separation")
def api_separation():
    from cleancut.background import check_runtime

    try:
        return jsonify(ok=True, **check_runtime())
    except Exception as exc:  # noqa: BLE001 -- optional runtime failures are user-facing diagnostics.
        return jsonify(ok=False, reason=str(exc))


@app.route("/api/job/<int:job_id>/speech/<int:index>", methods=["POST"])
def api_speech_preview(job_id: int, index: int):
    from cleancut.edl import EditDecisionList
    from cleancut.probe import pick_audio_track, probe_streams
    from cleancut.speech import eligible_words, prepare_replacements
    from cleancut.subtitles import read_srt

    job, path, err = _edl_job(job_id)
    if err:
        return err
    video = _video_for(job)
    if video is None or not video.is_file():
        return _bad("The source share is not mounted.", 404)
    edl = EditDecisionList.from_json(path)
    if not 0 <= index < len(edl.decisions):
        return _bad("No such decision.", 404)
    # Preview the word only, not the surrounding movie. Use the same cached
    # waveform at render time so an audition is not randomly generated again.
    selected = EditDecisionList(decisions=[edl.decisions[index], *edl.by_action("cut")])
    if not eligible_words(selected, []):
        return _bad("No accepted, word-precise profanity mute. Rescan older jobs first.")
    transcript = jobs.transcript_path(job_id)
    if not transcript.exists():
        return _bad("This scan has no saved reference transcript. Please scan again.")
    options = json.loads(job["options"] or "{}")
    try:
        track = pick_audio_track(probe_streams(video), options.get("audio_track"),
                                 prefer_language=options.get("prefer_language", "eng"))
        if track is None:
            return _bad("No usable audio track.")
        with tempfile.TemporaryDirectory(prefix="cleancut-preview-") as directory:
            clips = prepare_replacements(video, selected, read_srt(transcript), _speech_config(),
                                         Path(directory), audio_index=track.index,
                                         cache_dir=path.parent / "speech")
        if not clips:
            return _bad("Replacement unavailable: test the speech connection, or use a "
                        "3–12 second single-speaker reference. The word will stay muted.")
        return send_file(clips[0].path, mimetype="audio/wav", conditional=False)
    except Exception as exc:
        return _bad(f"Could not preview speech: {exc}")


@app.route("/api/job/<int:job_id>/background/<int:index>", methods=["POST"])
def api_background_preview(job_id: int, index: int):
    from cleancut.background import prepare_background, preview_mix
    from cleancut.config import Config
    from cleancut.editor_ranges import Range, normalize_cuts, shift_after_cuts
    from cleancut.edl import EditDecisionList
    from cleancut.probe import pick_audio_track, probe_duration, probe_streams
    from cleancut.speech import SpeechClip, prepare_replacements
    from cleancut.subtitles import read_srt

    job, path, err = _edl_job(job_id)
    if err:
        return err
    video = _video_for(job)
    if video is None or not video.is_file():
        return _bad("The source share is not mounted.", 404)
    edl = EditDecisionList.from_json(path)
    if not 0 <= index < len(edl.decisions):
        return _bad("No such decision.", 404)
    decision = edl.decisions[index]
    options = json.loads(job["options"] or "{}")
    try:
        track = pick_audio_track(probe_streams(video), options.get("audio_track"),
                                 prefer_language=options.get("prefer_language", "eng"))
        if track is None:
            return _bad("No usable audio track.")
        cuts = normalize_cuts([Range(d.start, d.end) for d in edl.by_action("cut")], probe_duration(video))
        target = Range(decision.start, decision.end)
        with tempfile.TemporaryDirectory(prefix="cleancut-background-preview-") as temporary:
            backgrounds = prepare_background(
                video, edl, Config(preserve_background=True), Path(temporary),
                audio_index=track.index, channels=track.channels or 2, cuts=cuts,
                cache_dir=path.parent / "background", only=target,
                language=options.get("prefer_language", "eng"),
            )
            if not backgrounds:
                return _bad("Background unavailable or rejected: only accepted word-timed mutes "
                            "outside cuts qualify. Check the render log and separation installation. "
                            "The full word mute remains in place.")
            # Preview source context; returned cache clips are on edited time.
            delta = target.start - shift_after_cuts(target.start, cuts)
            overlays = [SpeechClip(b.start + delta, b.end + delta, b.path) for b in backgrounds]
            transcript = jobs.transcript_path(job_id)
            if decision.action == "replace" and transcript.exists():
                selected = EditDecisionList(decisions=[decision, *edl.by_action("cut")])
                speech = prepare_replacements(video, selected, read_srt(transcript), _speech_config(),
                    Path(temporary) / "speech", audio_index=track.index,
                    cache_dir=path.parent / "speech", cuts=cuts)
                overlays.extend(SpeechClip(s.start + delta, s.end + delta, s.path) for s in speech)
            preview = path.parent / "background" / f"preview-{index}.wav"
            rendered = Path(temporary) / "preview.wav"
            preview_mix(video, target, rendered, audio_index=track.index, overlays=overlays)
            rendered.replace(preview)
        return send_file(preview, mimetype="audio/wav", conditional=False)
    except Exception as exc:  # noqa: BLE001 -- optional preview errors must not crash the web UI.
        return _bad(f"Could not preview background mix: {exc}")


@app.route("/api/health")
def api_health():
    return jsonify({
        "ok": True,
        "version": APP_VERSION,
        "media_roots": [str(r) for r in media_roots()],
        "output_dir": str(OUTPUT_DIR),
        "platform": platform.system(),
        "machine": platform.machine(),
        "native_macos": platform.system() == "Darwin",
    })


def main() -> None:
    ensure_dirs()
    jobs.init_db()
    integrations.init_db()
    jobs.start_worker()
    port = int(os.environ.get("PORT", "3000"))
    host = os.environ.get("HOST", "0.0.0.0")
    try:
        from waitress import serve
    except ImportError:
        app.run(host=host, port=port, threaded=True)
        return
    # Threads, not processes: the job worker is a thread in this process and
    # must not be forked into several competing copies.
    serve(app, host=host, port=port, threads=8, channel_timeout=1800)


if __name__ == "__main__":
    main()
