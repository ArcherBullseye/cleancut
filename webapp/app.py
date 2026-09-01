"""cleancut for Umbrel -- web front-end over the cleancut pipeline."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request, send_file

from webapp import jobs, library, review
from webapp import settings as settings_store
from webapp.paths import OUTPUT_DIR, ensure_dirs, media_roots

APP_VERSION = "1.0.6"

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


# --------------------------------------------------------------------------
# jobs
# --------------------------------------------------------------------------

@app.route("/api/jobs")
def api_jobs():
    return jsonify({"jobs": jobs.list_jobs()})


@app.route("/api/scan", methods=["POST"])
def api_scan():
    body = request.get_json(silent=True) or {}
    video = library.resolve_safe(body.get("path", ""))
    if video is None or not video.is_file():
        return _bad("That file is outside the mounted media roots.", 403)
    if not library.is_video(video):
        return _bad("Not a recognised video file.")

    existing = jobs.active_job_for(str(video))
    if existing:
        return jsonify({"ok": True, "job_id": existing["id"], "existing": True})

    cfg = settings_store.load()
    preset = body.get("preset") or cfg["preset"]
    if preset not in ("fast", "balanced", "thorough"):
        return _bad("Unknown preset.")

    options: dict[str, Any] = {
        "categories": body.get("categories") or cfg["categories"],
        "actions": body.get("actions") or cfg["actions"],
        "ollama_host": body.get("ollama_host", cfg["ollama_host"]),
        "llm_model": cfg["llm_model"],
        "vlm_model": cfg["vlm_model"],
        "prefer_language": body.get("prefer_language") or cfg["prefer_language"],
        "output_dir": cfg["output_dir"],
        "auto_render": bool(body.get("auto_render", cfg["auto_render"])),
        "use_visual": bool(body.get("use_visual", True)),
        "allow_solo_visual": bool(body.get("allow_solo_visual", False)),
    }
    for key in ("use_llm", "use_vlm", "use_audio_events"):
        if key in body:
            options[key] = bool(body[key])

    job_id = jobs.create_job(
        "scan", str(video), title=video.stem, preset=preset, options=options,
    )
    return jsonify({"ok": True, "job_id": job_id})


@app.route("/api/job/<int:job_id>")
def api_job(job_id: int):
    job, err = _job_or_404(job_id)
    if err:
        return err
    payload = dict(job)
    payload["options"] = json.loads(job["options"] or "{}")
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
    jobs.delete_job(job_id)
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
    render_id = jobs.queue_render(job_id, overrides=overrides)
    if render_id is None:
        return _bad("Could not queue the render.")
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
    return send_file(out, as_attachment=False, conditional=True,
                     download_name=out.name, mimetype="video/mp4")


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


@app.route("/api/health")
def api_health():
    return jsonify({
        "ok": True,
        "version": APP_VERSION,
        "media_roots": [str(r) for r in media_roots()],
        "output_dir": str(OUTPUT_DIR),
    })


def main() -> None:
    ensure_dirs()
    jobs.init_db()
    jobs.start_worker()
    port = int(os.environ.get("PORT", "3000"))
    try:
        from waitress import serve
    except ImportError:
        app.run(host="0.0.0.0", port=port, threaded=True)
        return
    # Threads, not processes: the job worker is a thread in this process and
    # must not be forked into several competing copies.
    serve(app, host="0.0.0.0", port=port, threads=8, channel_timeout=1800)


if __name__ == "__main__":
    main()
